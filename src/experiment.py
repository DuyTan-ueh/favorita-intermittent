"""Chạy thực nghiệm trả lời ba câu hỏi nghiên cứu.

  RQ1 — Two-Stage có vượt baseline cổ điển và mô hình một giai đoạn không?
  RQ2 — Đặc trưng khuyến mãi, lịch, sự kiện đóng góp bao nhiêu?
  RQ3 — Hiệu quả thay đổi thế nào giữa các nhóm mức độ gián đoạn?

Chạy::

    python -m src.experiment --config config/default.yaml --gap 0
    python -m src.experiment --config config/default.yaml --gap 7
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, models, significance
from .config import Config, load_config

KEYS = ["store_nbr", "item_nbr"]
NON_FEATURES = {"y", "y_occurrence", "y_magnitude", "date",
                "store_nbr", "item_nbr"}


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def load_features(cfg: Config, max_parts: int | None = None) -> pd.DataFrame:
    """Nạp dữ liệu đã sinh đặc trưng.

    Các tệp lô đã được xáo trộn từ bước trước nên đọc một phần vẫn cho mẫu đại
    diện — điều này cho phép giảm ``max_parts`` khi bộ nhớ eo hẹp mà không làm
    lệch kết quả.
    """
    parts = sorted(glob.glob(str(cfg.out_dir / "features" / "part_*.parquet")))
    if not parts:
        raise FileNotFoundError(
            f"Chưa có dữ liệu đặc trưng trong {cfg.out_dir}. "
            f"Chạy 'python -m src.pipeline' trước.")
    if max_parts:
        parts = parts[:max_parts]
        print(f"  [lưu ý] chỉ nạp {len(parts)} tệp lô để tiết kiệm bộ nhớ")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  nạp {len(df):,} dòng | {mb:.0f} MB")
    return df


def load_folds(cfg: Config, gap: int) -> list[dict]:
    path = cfg.out_dir / f"folds_gap{gap}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Kiểm tra 'gap_variants' trong cấu hình.")
    with open(path, encoding="utf-8") as fh:
        folds = json.load(fh)
    for f in folds:
        for k in ("train_start", "train_end", "test_start", "test_end"):
            f[k] = pd.Timestamp(f[k])
    return folds


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Chuyển cột phân loại sang dạng số để XGBoost xử lý được."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    for c in feature_cols:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes.astype("int16")
    return df, feature_cols


def run_fold(df: pd.DataFrame, fold: dict, feature_cols: list[str],
             specs: pd.DataFrame, cfg: Config, series: pd.DataFrame,
             run_baselines: bool = True) -> tuple[list[dict], list[pd.DataFrame]]:
    """Huấn luyện và đánh giá toàn bộ mô hình trên một fold.

    Trả về hai thứ tách bạch: bảng chỉ số tổng hợp, và bảng chỉ số phân tầng
    theo nhóm mẫu nhu cầu. Bảng thứ hai được lập cho CẢ Single-Stage lẫn
    Two-Stage, vì RQ3 hỏi khung hai giai đoạn có tốt hơn ở nhóm nào — muốn
    trả lời thì phải so hai mô hình trong cùng một nhóm, chứ một mình con số
    của Two-Stage không nói lên điều gì.
    """
    exp = cfg.raw.get("experiment", {})
    n_est = exp.get("n_estimators", 300)
    wanted_sets = exp.get("feature_sets") or list(models.FEATURE_SETS)
    stage2_list = exp.get("stage2_objectives") or ["squared"]
    shift = bool(exp.get("stage2_shift", False))
    device = models.resolve_device(exp.get("device", "auto"))
    save_losses = bool(exp.get("save_series_losses", True))

    train = df[df.date <= fold["train_end"]]
    test = df[(df.date >= fold["test_start"]) & (df.date <= fold["test_end"])]

    print(f"\n  fold {fold['fold']} | train {len(train):,} dòng "
          f"(tới {fold['train_end'].date()}) | test {len(test):,} dòng")

    scales = metrics.compute_scales(train)
    results, strat_frames, loss_frames = [], [], []

    # ---- Baseline cổ điển (RQ1) ----
    if run_baselines:
        for variant, label in [("classic", "Croston"), ("sba", "SBA"),
                               ("tsb", "TSB")]:
            t0 = time.time()
            pred = models.fit_predict_baseline(train, test, variant)
            res = metrics.evaluate_forecast(pred, scales)
            res.update({"model": label, "feature_set": "-",
                        "fold": fold["fold"], "gap": fold["gap_days"],
                        "seconds": round(time.time() - t0, 1)})
            results.append(res)

            # baseline cũng cần phân tầng để làm mốc so sánh trong RQ3
            s = metrics.evaluate_by_pattern(pred, series)
            s = s.assign(model=label, feature_set="-",
                         fold=fold["fold"], gap=fold["gap_days"])
            strat_frames.append(s)
            if save_losses:
                loss_frames.append(significance.per_series_losses(
                    pred, f"{label}|-", fold["fold"]))

            print(f"    {label:<26} WAPE {res['wape']:.4f} | "
                  f"{res['seconds']:.0f}s")

    # ---- Một giai đoạn và Two-Stage, cho từng bộ đặc trưng (RQ1 + RQ2) ----
    for set_name in wanted_sets:
        groups = models.FEATURE_SETS.get(set_name)
        if groups is None:
            print(f"    [bỏ qua] không có bộ đặc trưng '{set_name}'")
            continue
        cols = [c for c in models.select_features(specs, groups)
                if c in feature_cols]
        if not cols:
            continue

        # Với Two-Stage, mỗi hàm mất mát của giai đoạn hai là một biến thể
        # riêng. Chỉ chạy nhiều biến thể trên bộ đặc trưng đầy đủ để tránh
        # bùng nổ số lượt huấn luyện: mục đích là kiểm tra xem kết luận về
        # Two-Stage có phụ thuộc vào lựa chọn hàm mất mát hay không.
        variants = [("Single-Stage", models.fit_single_stage, {})]
        s2_here = stage2_list if set_name == "full" else stage2_list[:1]
        for s2 in s2_here:
            tag = "Two-Stage" if s2 == stage2_list[0] else f"Two-Stage[{s2}]"
            variants.append(("Two-Stage", models.fit_two_stage,
                             {"stage2": s2, "shift": shift, "_tag": tag}))

        for label, fn, extra in variants:
            tag = extra.pop("_tag", label)
            t0 = time.time()
            pred, _ = fn(train, test, cols, seed=cfg.seed,
                         n_estimators=n_est, device=device, **extra)
            res = metrics.evaluate_forecast(pred, scales)

            if label == "Two-Stage":
                res.update(metrics.occurrence_metrics(
                    pred.y_occurrence.to_numpy(),
                    pred.prob_occurrence.to_numpy()))
                res["stage2"] = extra.get("stage2", "squared")

            res.update({"model": tag, "feature_set": set_name,
                        "fold": fold["fold"], "gap": fold["gap_days"],
                        "n_features": len(cols), "device": device,
                        "seconds": round(time.time() - t0, 1)})
            results.append(res)

            s = metrics.evaluate_by_pattern(pred, series)
            s = s.assign(model=tag, feature_set=set_name,
                         fold=fold["fold"], gap=fold["gap_days"])
            strat_frames.append(s)

            if save_losses:
                loss_frames.append(significance.per_series_losses(
                    pred, f"{tag}|{set_name}", fold["fold"]))

            print(f"    {tag + ' [' + set_name + ']':<32} "
                  f"WAPE {res['wape']:.4f} | {res['seconds']:.0f}s")
            del pred
            gc.collect()

    return results, strat_frames, loss_frames


def run_experiment(cfg: Config, gap: int, max_parts: int | None = None,
                   quick: bool = False) -> pd.DataFrame:
    _banner(f"THỰC NGHIỆM — gap = {gap} ngày")

    specs = pd.read_csv(cfg.out_dir / "feature_specs.csv")
    series = pd.read_parquet(cfg.out_dir / "series_selected.parquet")
    folds = load_folds(cfg, gap)

    out_dir = cfg.out_dir / f"results_gap{gap}"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    use_ckpt = cfg.raw.get("experiment", {}).get("checkpoint", True)
    if use_ckpt:
        ckpt_dir.mkdir(exist_ok=True)

    if quick:
        folds = folds[-1:]
        print("  [chế độ nhanh] chỉ chạy fold cuối")

    # Bỏ qua những fold đã hoàn tất ở lần chạy trước. Việc này quan trọng vì
    # một lượt chạy đầy đủ mất nhiều giờ và phiên làm việc có thể bị ngắt.
    todo = []
    for f in folds:
        done = ckpt_dir / f"fold{f['fold']}_metrics.csv"
        if use_ckpt and done.exists():
            print(f"  [bỏ qua] fold {f['fold']} đã có kết quả từ lần chạy trước")
        else:
            todo.append(f)

    if todo:
        df = load_features(cfg, max_parts)
        df, feature_cols = prepare_matrix(df)
    else:
        print("  Mọi fold đã hoàn tất, chỉ tổng hợp lại kết quả")

    for fold in todo:
        res, strat, losses = run_fold(df, fold, feature_cols, specs, cfg,
                                      series)
        if use_ckpt:
            pd.DataFrame(res).to_csv(
                ckpt_dir / f"fold{fold['fold']}_metrics.csv", index=False)
            pd.concat(strat, ignore_index=True).to_csv(
                ckpt_dir / f"fold{fold['fold']}_pattern.csv", index=False)
            if losses:
                pd.concat(losses, ignore_index=True).to_parquet(
                    ckpt_dir / f"fold{fold['fold']}_losses.parquet",
                    index=False)
            print(f"  [đã lưu] checkpoint fold {fold['fold']}")
        gc.collect()

    # Gộp toàn bộ checkpoint lại, kể cả của những lần chạy trước
    metric_files = sorted(ckpt_dir.glob("fold*_metrics.csv")) if use_ckpt else []
    pattern_files = sorted(ckpt_dir.glob("fold*_pattern.csv")) if use_ckpt else []
    if not metric_files:
        raise RuntimeError("Không có kết quả nào để tổng hợp")

    results = pd.concat([pd.read_csv(f) for f in metric_files],
                        ignore_index=True)
    strat = (pd.concat([pd.read_csv(f) for f in pattern_files],
                       ignore_index=True) if pattern_files else None)

    results.to_csv(out_dir / "metrics_by_fold.csv", index=False)
    if strat is not None:
        strat.to_csv(out_dir / "metrics_by_pattern.csv", index=False)

    loss_files = sorted(ckpt_dir.glob("fold*_losses.parquet")) if use_ckpt else []
    losses = (pd.concat([pd.read_parquet(f) for f in loss_files],
                        ignore_index=True) if loss_files else None)

    _summarise(results, strat, out_dir, gap)
    if losses is not None:
        _run_significance(losses, results, out_dir, gap)
    return results


def _summarise(results: pd.DataFrame, strat: pd.DataFrame | None,
               out_dir: Path, gap: int) -> None:
    """In các bảng tổng hợp tương ứng từng câu hỏi nghiên cứu."""
    agg = (results.groupby(["model", "feature_set"], as_index=False)
           .agg(wape=("wape", "mean"), mae=("mae", "mean"),
                rmse=("rmse", "mean"), rmsse=("rmsse", "mean"),
                seconds=("seconds", "mean"))
           .sort_values("wape"))
    agg.to_csv(out_dir / "summary.csv", index=False)

    _banner(f"RQ1 — SO SÁNH MÔ HÌNH (gap={gap}, trung bình các fold)")
    print(agg.round(4).to_string(index=False))

    # ---------------- RQ2 ----------------
    _banner(f"RQ2 — ĐÓNG GÓP CỦA TỪNG NHÓM ĐẶC TRƯNG (gap={gap})")
    order = [s for s in models.FEATURE_SETS
             if s in set(results.feature_set)]

    for model_name in ["Single-Stage", "Two-Stage"]:
        sub = results[results.model == model_name]
        if not len(sub):
            continue
        ab = (sub.groupby("feature_set", as_index=False)
              .agg(wape=("wape", "mean"), rmse=("rmse", "mean")))
        ab["_o"] = ab.feature_set.map({k: i for i, k in enumerate(order)})
        ab = ab.sort_values("_o").drop(columns="_o").reset_index(drop=True)

        base = ab.wape.iloc[0]
        ab["tích_luỹ_%"] = ((base - ab.wape) / base * 100).round(2)
        # Đóng góp RIÊNG của nhóm vừa thêm vào ở mỗi bước
        ab["riêng_%"] = ab["tích_luỹ_%"].diff().fillna(0).round(2)

        print(f"\n[{model_name}]")
        print(ab.round(4).to_string(index=False))

    print("\n  Đọc bảng: mỗi dòng thêm đúng một nhóm đặc trưng so với dòng "
          "trên,\n  nên cột 'riêng_%' là đóng góp của riêng nhóm đó.")
    print("  Dòng 'hist_cal_hol_promo' cô lập đóng góp của KHUYẾN MÃI.")

    occ = results[(results.model == "Two-Stage")].dropna(subset=["pr_auc"])
    if len(occ):
        print("\n  Giai đoạn 1 — nhận biết ngày có đơn:")
        print(occ.groupby("feature_set")[
            ["precision", "recall", "f1", "pr_auc"]]
            .mean().round(4).to_string())

    # ---------------- RQ3 ----------------
    if strat is None or not len(strat):
        return

    _banner(f"RQ3 — HIỆU QUẢ THEO NHÓM MẪU NHU CẦU (gap={gap})")
    full = strat[strat.feature_set.isin(["full", "-"])]

    pivot = (full.pivot_table(index="pattern", columns="model",
                              values="wape", aggfunc="mean").round(4))
    cols = [c for c in ["Croston", "SBA", "TSB",
                        "Single-Stage", "Two-Stage"] if c in pivot.columns]
    pivot = pivot[cols]

    if {"Single-Stage", "Two-Stage"}.issubset(pivot.columns):
        # Số âm nghĩa là Two-Stage tốt hơn (WAPE thấp hơn)
        pivot["Two−Single"] = (pivot["Two-Stage"]
                               - pivot["Single-Stage"]).round(4)
        pivot["Two thắng?"] = np.where(pivot["Two−Single"] < 0, "có", "không")

    info = (full[full.model == "Two-Stage"]
            .groupby("pattern")[["n_series", "zero_rate"]].mean().round(4))
    out = info.join(pivot)
    order_p = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    out = out.reindex([p for p in order_p if p in out.index])

    print(out.to_string())
    print("\n  Cột 'Two−Single' âm nghĩa là khung hai giai đoạn tốt hơn ở "
          "nhóm đó.\n  Đây chính là câu trả lời cho RQ3.")
    out.to_csv(out_dir / "rq3_model_by_pattern.csv")

    print(f"\nKết quả lưu tại: {out_dir}")


def _run_significance(losses: pd.DataFrame, results: pd.DataFrame,
                      out_dir: Path, gap: int) -> None:
    """Kiểm định xem chênh lệch giữa các mô hình có ý nghĩa thống kê không.

    Chênh lệch WAPE trong nghiên cứu này chỉ vào khoảng vài phần nghìn. Một
    bảng số trung bình không đủ để khẳng định đó là khác biệt thật, nên phần
    này đối chiếu từng mô hình với mô hình tốt nhất bằng kiểm định ghép cặp
    trên từng chuỗi, kèm hiệu chỉnh cho việc so sánh nhiều lần.
    """
    _banner(f"KIỂM ĐỊNH THỐNG KÊ (gap={gap})")

    best = (results.groupby(["model", "feature_set"])["wape"].mean()
            .idxmin())
    reference = f"{best[0]}|{best[1]}"
    print(f"  Mô hình tham chiếu: {reference}\n")

    table = significance.compare_all(losses, reference)
    if not len(table):
        print("  Không đủ dữ liệu để kiểm định")
        return

    show = table[["model_b", "n_series", "mean_diff", "b_win_rate",
                  "p_wilcoxon", "ý_nghĩa"]].rename(columns={
        "model_b": "so_với", "mean_diff": "chênh_MAE",
        "b_win_rate": "tỷ_lệ_thắng"})
    print(show.round(5).to_string(index=False))
    print("\n  chênh_MAE âm nghĩa là mô hình tham chiếu tốt hơn.")
    print("  tỷ_lệ_thắng là tỷ lệ chuỗi mà mô hình kia thắng tham chiếu.")
    print("  Cột ý_nghĩa đã hiệu chỉnh Holm cho nhiều phép so sánh.")
    table.to_csv(out_dir / "significance_vs_best.csv", index=False)

    # So sánh trực tiếp cặp quan trọng nhất của RQ1
    pooled = (losses.groupby(["model_key", "store_nbr", "item_nbr"],
                             as_index=False).agg(mae=("mae", "mean")))
    pair = [("Single-Stage|full", "Two-Stage|full")]
    rows = [significance.paired_test_by_series(pooled, a, b) for a, b in pair
            if a in set(pooled.model_key) and b in set(pooled.model_key)]
    if rows:
        print("\n  RQ1 — so sánh trực tiếp Single-Stage và Two-Stage:")
        for r in rows:
            print(f"    {r['model_a']} vs {r['model_b']}")
            print(f"      chênh MAE trung bình : {r['mean_diff']:+.5f}")
            print(f"      tỷ lệ chuỗi B thắng  : {r['b_win_rate']:.1%}")
            print(f"      p (t-test)           : {r['p_ttest']:.3e}")
            print(f"      p (Wilcoxon)         : {r['p_wilcoxon']:.3e}")
            print(f"      kết luận             : {r['verdict']}")
        pd.DataFrame(rows).to_csv(out_dir / "significance_rq1.csv", index=False)

    print(f"\n  Đã lưu kết quả kiểm định vào {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Thực nghiệm mô hình")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--gap", type=int, default=None,
                    help="chạy một giá trị gap; bỏ trống = chạy tất cả")
    ap.add_argument("--max-parts", type=int, default=None,
                    help="giới hạn số tệp lô nạp vào, dùng khi thiếu bộ nhớ")
    ap.add_argument("--quick", action="store_true",
                    help="chỉ chạy fold cuối để kiểm thử nhanh")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gaps = ([args.gap] if args.gap is not None
            else cfg.split.get("gap_variants", [cfg.split["gap_days"]]))

    frames = []
    for gap in gaps:
        frames.append(run_experiment(cfg, gap, args.max_parts, args.quick))

    if len(frames) > 1:
        _banner("SO SÁNH GIỮA CÁC GIÁ TRỊ GAP")
        allr = pd.concat(frames, ignore_index=True)
        pivot = allr.pivot_table(index=["model", "feature_set"],
                                 columns="gap", values="wape",
                                 aggfunc="mean").round(4)
        pivot.columns = [f"gap={c}" for c in pivot.columns]
        if pivot.shape[1] == 2:
            a, b = pivot.columns
            pivot["chênh_lệch"] = (pivot[b] - pivot[a]).round(4)
        print(pivot.to_string())
        pivot.to_csv(cfg.out_dir / "gap_comparison.csv")
        print(f"\nĐã lưu: {cfg.out_dir / 'gap_comparison.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
