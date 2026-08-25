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

from . import metrics, models
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
             specs: pd.DataFrame, cfg: Config,
             run_baselines: bool = True) -> list[dict]:
    """Huấn luyện và đánh giá toàn bộ mô hình trên một fold."""
    train = df[df.date <= fold["train_end"]]
    test = df[(df.date >= fold["test_start"]) & (df.date <= fold["test_end"])]

    print(f"\n  fold {fold['fold']} | train {len(train):,} dòng "
          f"(tới {fold['train_end'].date()}) | test {len(test):,} dòng")

    scales = metrics.compute_scales(train)
    results = []

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
            print(f"    {label:<22} WAPE {res['wape']:.4f} | "
                  f"{res['seconds']:.0f}s")

    # ---- Một giai đoạn và Two-Stage, cho từng bộ đặc trưng (RQ1 + RQ2) ----
    for set_name, groups in models.FEATURE_SETS.items():
        cols = [c for c in models.select_features(specs, groups)
                if c in feature_cols]
        if not cols:
            continue

        for label, fn in [("Single-Stage", models.fit_single_stage),
                          ("Two-Stage", models.fit_two_stage)]:
            t0 = time.time()
            pred, _ = fn(train, test, cols, seed=cfg.seed)
            res = metrics.evaluate_forecast(pred, scales)

            if label == "Two-Stage":
                res.update(metrics.occurrence_metrics(
                    pred.y_occurrence.to_numpy(),
                    pred.prob_occurrence.to_numpy()))

            res.update({"model": label, "feature_set": set_name,
                        "fold": fold["fold"], "gap": fold["gap_days"],
                        "n_features": len(cols),
                        "seconds": round(time.time() - t0, 1)})
            results.append(res)
            print(f"    {label + ' [' + set_name + ']':<22} "
                  f"WAPE {res['wape']:.4f} | {res['seconds']:.0f}s")

            # Phân tầng theo nhóm nhu cầu cho cấu hình đầy đủ (RQ3)
            if set_name == "full":
                res["_stratified"] = pred
        gc.collect()

    return results


def run_experiment(cfg: Config, gap: int, max_parts: int | None = None,
                   quick: bool = False) -> pd.DataFrame:
    _banner(f"THỰC NGHIỆM — gap = {gap} ngày")

    specs = pd.read_csv(cfg.out_dir / "feature_specs.csv")
    series = pd.read_parquet(cfg.out_dir / "series_selected.parquet")
    folds = load_folds(cfg, gap)

    df = load_features(cfg, max_parts)
    df, feature_cols = prepare_matrix(df)

    if quick:
        folds = folds[-1:]
        print("  [chế độ nhanh] chỉ chạy fold cuối")

    all_results, strat_frames = [], []
    for fold in folds:
        res = run_fold(df, fold, feature_cols, specs, cfg)
        for r in res:
            pred = r.pop("_stratified", None)
            if pred is not None and r["model"] == "Two-Stage":
                s = metrics.evaluate_by_pattern(pred, series)
                s["fold"] = fold["fold"]
                s["gap"] = gap
                strat_frames.append(s)
        all_results.extend(res)
        gc.collect()

    results = pd.DataFrame(all_results)
    out_dir = cfg.out_dir / f"results_gap{gap}"
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / "metrics_by_fold.csv", index=False)

    if strat_frames:
        strat = pd.concat(strat_frames, ignore_index=True)
        strat.to_csv(out_dir / "metrics_by_pattern.csv", index=False)

    _summarise(results, strat_frames, out_dir, gap)
    return results


def _summarise(results: pd.DataFrame, strat_frames: list,
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

    two = results[results.model == "Two-Stage"]
    if len(two):
        _banner(f"RQ2 — ĐÓNG GÓP CỦA TỪNG NHÓM ĐẶC TRƯNG (gap={gap})")
        ab = (two.groupby("feature_set", as_index=False)
              .agg(wape=("wape", "mean"), rmse=("rmse", "mean")))
        order = list(models.FEATURE_SETS)
        ab["_o"] = ab.feature_set.map({k: i for i, k in enumerate(order)})
        ab = ab.sort_values("_o").drop(columns="_o")
        base = ab.wape.iloc[0]
        ab["cải_thiện_%"] = ((base - ab.wape) / base * 100).round(2)
        print(ab.round(4).to_string(index=False))

        occ = two.dropna(subset=["pr_auc"])
        if len(occ):
            print("\n  Giai đoạn 1 — nhận biết ngày có đơn:")
            print(occ.groupby("feature_set")[
                ["precision", "recall", "f1", "pr_auc"]]
                .mean().round(4).to_string())

    if strat_frames:
        _banner(f"RQ3 — HIỆU QUẢ THEO NHÓM MẪU NHU CẦU (gap={gap})")
        strat = pd.concat(strat_frames, ignore_index=True)
        print(strat.groupby("pattern")[
            ["n_series", "zero_rate", "wape", "mae", "rmse"]]
            .mean().round(4).to_string())

    print(f"\nKết quả lưu tại: {out_dir}")


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
