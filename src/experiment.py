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

# --------------------------------------------------------------------------- #
# Cặp so sánh CỐ ĐỊNH của RQ3
# --------------------------------------------------------------------------- #
# Cặp này phải KHỚP CHÍNH XÁC hàm mất mát, nếu không RQ3 sẽ không trả lời được
# câu hỏi nó tự đặt ra.
#
# Bản trước dùng ("Single-Stage", "Two-Stage") — hai nhãn mặc định. Cặp đó cố
# định thật (khai báo trước khi xem kết quả test), nhưng "Single-Stage" mặc
# định là Tweedie còn "Two-Stage" mặc định là squared, nên kiến trúc VÀ hàm
# mất mát thay đổi cùng lúc. Chênh lệch quan sát được khi đó không quy về
# riêng kiến trúc được — đúng thứ mà ``_matched_loss_comparison`` đã cảnh báo
# ở phần RQ1, tạo ra mâu thuẫn nội bộ giữa hai mục của cùng một báo cáo.
#
# Chọn squared làm cặp chính (thay vì absolute) vì: độ chệch của cả hai mô
# hình đều gần 1, nên hiệu ứng không bị bóp méo bởi hiện tượng dự báo thiếu
# nặng vốn thấy ở nhóm absolute; và đây cũng chính là cặp đã được kiểm tra độ
# ổn định qua nhiều seed ở ``_seed_stability``.
#
# LƯU Ý về quy ước đặt tên: ``run_fold`` gán nhãn ngắn "Two-Stage" cho phần tử
# ĐẦU TIÊN của ``experiment.stage2_objectives`` (mặc định là ``squared``), và
# "Single-Stage[squared]" cho biến thể squared của mô hình một giai đoạn. Nếu
# đổi thứ tự ``stage2_objectives`` trong cấu hình thì nhãn "Two-Stage" sẽ trỏ
# sang hàm mất mát khác và cặp dưới đây KHÔNG còn khớp — ``_assert_rq3_pair_
# matched`` kiểm tra đúng tình huống đó và cảnh báo.
RQ3_FIXED_PAIR = ("Single-Stage[squared]", "Two-Stage")

# Giá trị p nhỏ hơn ngưỡng này bị làm tròn thành 0.0 do giới hạn số học dấu
# phẩy động. In "p = 0" là sai về bản chất: xác suất không bằng không, chỉ là
# nhỏ hơn mức biểu diễn được. Báo cáo dạng chặn trên thay vì con số bịa.
P_UNDERFLOW = 1e-300


def _fmt_p(p: float | None) -> str:
    """Định dạng giá trị p, không bao giờ in 'p = 0'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p <= 0.0:
        return f"<{P_UNDERFLOW:.0e}"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


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
             device: str = "cpu", seed: int | None = None,
             run_baselines: bool = True) -> tuple[list[dict], list[pd.DataFrame], list]:
    """Huấn luyện và đánh giá toàn bộ mô hình trên một fold, tại một seed.

    Tham số ``seed`` điều khiển tính ngẫu nhiên của XGBoost (``subsample`` và
    ``colsample_bytree`` đều nhỏ hơn 1). Baseline cổ điển không có tính ngẫu
    nhiên nên tham số này không ảnh hưởng tới chúng — gọi hàm với
    ``run_baselines=False`` ở các seed sau seed đầu để tránh tính lại vô ích.

    Trả về ba thứ tách bạch: bảng chỉ số tổng hợp, bảng chỉ số phân tầng theo
    nhóm mẫu nhu cầu, và danh sách sai số theo từng chuỗi.
    """
    exp = cfg.raw.get("experiment", {})
    n_est = exp.get("n_estimators", 300)
    wanted_sets = exp.get("feature_sets") or list(models.FEATURE_SETS)
    s2_list = exp.get("stage2_objectives") or ["squared"]
    s1_list = exp.get("single_stage_objectives") or ["tweedie"]
    shift = bool(exp.get("stage2_shift", False))
    xgb_overrides = exp.get("xgb_params") or None
    save_losses = bool(exp.get("save_series_losses", True))
    seed = cfg.seed if seed is None else seed

    train = df[df.date <= fold["train_end"]]
    test = df[(df.date >= fold["test_start"]) & (df.date <= fold["test_end"])]

    print(f"\n  fold {fold['fold']} seed {seed} | train {len(train):,} dòng "
          f"(tới {fold['train_end'].date()}) | test {len(test):,} dòng")

    scales = metrics.compute_scales(train)
    results, strat_frames, loss_frames = [], [], []

    # ---- Baseline cổ điển (RQ1) ----
    # Không có tính ngẫu nhiên nên chỉ cần tính một lần, không phụ thuộc seed.
    if run_baselines:
        for variant, label in [("classic", "Croston"), ("sba", "SBA"),
                               ("tsb", "TSB")]:
            t0 = time.time()
            pred = models.fit_predict_baseline(train, test, variant)
            res = metrics.evaluate_forecast(pred, scales)
            res.update({"model": label, "feature_set": "-",
                        "fold": fold["fold"], "gap": fold["gap_days"],
                        "seed": seed,
                        "seconds": round(time.time() - t0, 1)})
            results.append(res)

            # baseline cũng cần phân tầng để làm mốc so sánh trong RQ3
            s = metrics.evaluate_by_pattern(pred, series)
            s = s.assign(model=label, feature_set="-",
                         fold=fold["fold"], gap=fold["gap_days"], seed=seed)
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

        # Thiết kế giai thừa trên bộ đặc trưng đầy đủ.
        #
        # Kiến trúc và hàm mất mát là hai yếu tố tách biệt. Nếu chỉ khung hai
        # giai đoạn được thử nhiều hàm mất mát trong khi mô hình một giai đoạn
        # cố định ở Tweedie, thì phần chênh lệch quan sát được sẽ lẫn giữa ảnh
        # hưởng của kiến trúc và ảnh hưởng của hàm mất mát, và không thể quy
        # kết cho yếu tố nào.
        #
        # Ở các bộ đặc trưng khác chỉ chạy cấu hình mặc định, vì mục đích của
        # chúng là nghiên cứu loại trừ đặc trưng chứ không phải so kiến trúc.
        full_set = (set_name == "full")
        s1_here = s1_list if full_set else s1_list[:1]
        s2_here = s2_list if full_set else s2_list[:1]

        variants = []
        for obj in s1_here:
            tag = ("Single-Stage" if obj == s1_list[0]
                   else f"Single-Stage[{obj}]")
            variants.append(("Single-Stage", models.fit_single_stage,
                             {"objective": obj}, tag))
        for obj in s2_here:
            tag = "Two-Stage" if obj == s2_list[0] else f"Two-Stage[{obj}]"
            variants.append(("Two-Stage", models.fit_two_stage,
                             {"stage2": obj, "shift": shift}, tag))

        for label, fn, extra, tag in variants:
            t0 = time.time()
            pred, _ = fn(train, test, cols, seed=seed,
                         n_estimators=n_est, device=device,
                         xgb_overrides=xgb_overrides, **extra)
            res = metrics.evaluate_forecast(pred, scales)

            if label == "Two-Stage":
                res.update(metrics.occurrence_metrics(
                    pred.y_occurrence.to_numpy(),
                    pred.prob_occurrence.to_numpy()))

            res.update({
                "model": tag, "arch": label, "feature_set": set_name,
                "loss": extra.get("stage2") or extra.get("objective"),
                "fold": fold["fold"], "gap": fold["gap_days"], "seed": seed,
                "n_features": len(cols), "device": device,
                "seconds": round(time.time() - t0, 1)})
            results.append(res)

            s = metrics.evaluate_by_pattern(pred, series)
            s = s.assign(model=tag, arch=label, feature_set=set_name,
                         fold=fold["fold"], gap=fold["gap_days"], seed=seed)
            strat_frames.append(s)

            if save_losses:
                loss_frames.append(significance.per_series_losses(
                    pred, f"{tag}|{set_name}", fold["fold"]))

            print(f"    {tag + ' [' + set_name + ']':<34} "
                  f"WAPE {res['wape']:.4f} | bias {res['bias_ratio']:.3f} | "
                  f"{res['seconds']:.0f}s")
            del pred
            gc.collect()

    return results, strat_frames, loss_frames


def run_experiment(cfg: Config, gap: int, max_parts: int | None = None,
                   quick: bool = False,
                   summarize_only: bool = False) -> pd.DataFrame:
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

    device = models.resolve_device(
        cfg.raw.get("experiment", {}).get("device", "auto"))

    seeds = cfg.raw.get("run", {}).get("seeds") or [cfg.seed]
    if len(seeds) > 1:
        print(f"  Chạy đa seed: {seeds}")

    if quick:
        folds = folds[-1:]
        print("  [chế độ nhanh] chỉ chạy fold cuối")

    # Bỏ qua những tổ hợp (fold, seed) đã hoàn tất ở lần chạy trước. Việc này
    # quan trọng vì một lượt chạy đầy đủ mất nhiều giờ và phiên làm việc có
    # thể bị ngắt.
    todo = []
    for f in folds:
        for s in seeds:
            done = ckpt_dir / f"fold{f['fold']}_seed{s}_metrics.csv"
            if use_ckpt and done.exists():
                print(f"  [bỏ qua] fold {f['fold']} seed {s} đã có kết quả")
            else:
                todo.append((f, s))

    if summarize_only:
        if todo:
            raise RuntimeError(
                f"Còn {len(todo)} tổ hợp (fold, seed) chưa có checkpoint, "
                f"không thể chỉ tổng hợp. Bỏ cờ --summarize-only để chạy chúng.")
        print("  [chỉ tổng hợp] dùng lại checkpoint, không huấn luyện lại")
        todo = []
    elif todo:
        df = load_features(cfg, max_parts)
        df, feature_cols = prepare_matrix(df)
    else:
        print("  Mọi tổ hợp đã hoàn tất, chỉ tổng hợp lại kết quả")

    for fold, seed in todo:
        # Baseline cổ điển không có tính ngẫu nhiên: chỉ tính ở seed đầu tiên
        # trong danh sách, tránh lặp lại vô ích cho các seed sau.
        run_baselines = (seed == seeds[0])
        res, strat, losses = run_fold(df, fold, feature_cols, specs, cfg,
                                      series, device=device, seed=seed,
                                      run_baselines=run_baselines)
        if use_ckpt:
            pd.DataFrame(res).to_csv(
                ckpt_dir / f"fold{fold['fold']}_seed{seed}_metrics.csv",
                index=False)
            pd.concat(strat, ignore_index=True).to_csv(
                ckpt_dir / f"fold{fold['fold']}_seed{seed}_pattern.csv",
                index=False)
            if losses:
                pd.concat(losses, ignore_index=True).to_parquet(
                    ckpt_dir / f"fold{fold['fold']}_seed{seed}_losses.parquet",
                    index=False)
            print(f"  [đã lưu] checkpoint fold {fold['fold']} seed {seed}")
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
        _run_significance(losses, results, out_dir, gap, series,
                          getattr(_summarise, "best_pair", None))
    return results


def _bias_diagnostics(results: pd.DataFrame, gap: int) -> None:
    """Kiểm tra tính lành mạnh của dự báo, không chỉ độ chính xác.

    Một mô hình có thể đạt sai số nhỏ mà vẫn không dùng được. Trường hợp điển
    hình là huấn luyện bằng sai số tuyệt đối trên chuỗi có quá nửa số ngày
    không phát sinh nhu cầu: đại lượng mà sai số tuyệt đối tối thiểu hoá là
    trung vị có điều kiện, và trung vị khi đó bằng đúng không. Mô hình học được
    cách dự báo bằng không cho gần như mọi ngày, đạt MAE rất tốt, nhưng gây
    hết hàng liên tục nếu đưa vào vận hành.

    Bảng này phơi bày hiện tượng đó qua hai con số: tỷ số tổng dự báo trên tổng
    nhu cầu thực, và tỷ lệ dự báo gần bằng không.
    """
    if "bias_ratio" not in results.columns:
        return
    _banner(f"CHẨN ĐOÁN ĐỘ CHỆCH (gap={gap})")

    tab = (results[results.feature_set.isin(["full", "-"])]
           .groupby("model", as_index=False)
           .agg(wape=("wape", "mean"), bias=("bias_ratio", "mean"),
                near_zero=("near_zero_rate", "mean"))
           .sort_values("wape"))

    # Ngưỡng 5%: với bài toán tồn kho, lệch 10% đã đủ gây hết hàng hoặc ứ
    # đọng trên diện rộng. Ngưỡng lỏng hơn sẽ bỏ lọt đúng những mô hình đang
    # đạt điểm số tốt nhờ dự báo thấp.
    tab["đánh_giá"] = np.where(
        tab.bias < 0.95, "dự báo THIẾU",
        np.where(tab.bias > 1.05, "dự báo THỪA", "cân bằng"))

    print(tab.round(4).to_string(index=False))
    print("\n  bias = tổng dự báo / tổng nhu cầu thực; bằng 1 là cân bằng.")
    print("  near_zero = tỷ lệ dự báo nhỏ hơn 0,5 đơn vị.")
    print("  Với bài toán tồn kho, dự báo thiếu hệ thống gây hết hàng, "
          "nên nguy hiểm\n  hơn sai số ngẫu nhiên cùng độ lớn.")

    bad = tab[tab.bias < 0.95]
    if len(bad):
        print(f"\n  [CẢNH BÁO] {len(bad)} mô hình dự báo thiếu: "
              f"{', '.join(bad.model)}")

    # Kiểm tra xem thứ hạng theo WAPE có bị chi phối bởi độ chệch không.
    # Nếu hai đại lượng tương quan mạnh, mô hình đứng đầu bảng WAPE đang đạt
    # điểm cao một phần nhờ dự báo thấp chứ không phải nhờ dự báo đúng hơn.
    if len(tab) >= 4:
        corr = float(tab.wape.corr(tab.bias))
        print(f"\n  Tương quan giữa WAPE và độ chệch: {corr:+.3f}")
        if corr > 0.6:
            print("  [QUAN TRỌNG] Tương quan dương mạnh giữa WAPE và độ "
                  "chệch trên dữ liệu này:\n"
                  "  mô hình có WAPE thấp hơn có xu hướng dự báo thiếu nhiều "
                  "hơn. Đây là hệ quả\n"
                  "  của việc sai số tuyệt đối tối thiểu hoá tại trung vị, "
                  "vốn thấp hơn kỳ vọng\n"
                  "  trên dữ liệu nhiều số không (xem Kolassa 2016). Không "
                  "nên xếp hạng mô hình\n"
                  "  chỉ bằng WAPE trên dữ liệu này — phải đối chiếu với "
                  "RMSE, RMSSE và độ chệch\n"
                  "  trước khi kết luận.")


def _matched_loss_comparison(results: pd.DataFrame, out_dir: Path,
                             gap: int) -> None:
    """So sánh kiến trúc theo hai mức độ khớp hàm mất mát, tách biệt rõ ràng.

    So một mô hình huấn luyện bằng sai số tuyệt đối với một mô hình huấn luyện
    bằng Tweedie, rồi đánh giá bằng WAPE, là phép so lệch: chỉ số đánh giá dựa
    trên sai số tuyệt đối nên đương nhiên ưu ái mô hình đã tối ưu đúng đại
    lượng đó.

    Có hai mức độ "khớp" cần phân biệt rạch ròi, không được gộp chung:

      Khớp chính xác (exact match)
          Hai kiến trúc dùng ĐÚNG cùng một hàm mất mát
          (``squared`` với ``squared``, ``absolute`` với ``absolute``).
          Hàm mất mát được giữ cố định, nên cách so sánh này cô lập ảnh
          hưởng của kiến trúc chặt hơn mọi cặp khác trong nghiên cứu.

          Cần nói đúng mức: điều này KHÔNG có nghĩa mọi chênh lệch quan sát
          được đều quy về kiến trúc. Hai nhánh vẫn khác nhau ở những điểm
          khác — số lượng mô hình được huấn luyện, tập quan sát mà từng mô
          hình nhìn thấy (giai đoạn hai chỉ học trên y>0), cách tính dự báo
          cuối, và nhiễu ngẫu nhiên của quá trình huấn luyện. Diễn đạt an
          toàn là "giữ cố định hàm mất mát giúp cô lập ảnh hưởng của kiến
          trúc chặt hơn", không phải "chênh lệch hoàn toàn do kiến trúc".

      Tương đồng phân phối (distributional analogy), KHÔNG phải khớp chính xác
          Tweedie (Single-Stage) và Gamma (Two-Stage) là hai hàm mất mát
          KHÁC NHAU — Tweedie có khối xác suất tại không, Gamma yêu cầu mục
          tiêu dương ngặt và chỉ áp dụng được vì giai đoạn hai đã lọc sẵn
          y>0. Cả hai chỉ giống nhau ở việc cùng được chọn vì phù hợp với
          đặc tính lệch phải của phần dữ liệu chúng xử lý. So sánh cặp này
          trả lời một câu hỏi khác: "khi mỗi kiến trúc dùng hàm mất mát phù
          hợp nhất với phần dữ liệu nó xử lý, kiến trúc nào tốt hơn?" —
          không phải "kiến trúc nào tốt hơn khi hàm mất mát giống hệt nhau?"

    Gộp hai câu hỏi này vào một bảng, hoặc gọi cả hai là "matched loss", sẽ
    khiến người đọc hiểu nhầm mức độ chặt chẽ của so sánh.
    """
    if "arch" not in results.columns or "loss" not in results.columns:
        return

    full = results[results.feature_set == "full"]
    if not len(full):
        return

    _banner(f"SO SÁNH THEO MỨC ĐỘ KHỚP HÀM MẤT MÁT (gap={gap})")

    def _show(label: str, sub: pd.DataFrame, metric: str) -> dict | None:
        if not len(sub):
            return None
        agg = (sub.groupby(["arch", "loss"], as_index=False)
               .agg(**{metric: (metric, "mean")}, bias=("bias_ratio", "mean"))
               .sort_values(metric))
        print(f"\n[{label}]")
        print(agg.round(4).to_string(index=False))
        return {"category": label, "metric": metric,
               "best_arch": agg.iloc[0]["arch"],
               "best_loss": agg.iloc[0]["loss"],
               "best_value": float(agg.iloc[0][metric])}

    rows = []
    r = _show("KHỚP CHÍNH XÁC — squared vs squared",
              full[(full.loss == "squared")], "wape")
    if r:
        rows.append(r)
    r = _show("KHỚP CHÍNH XÁC — absolute vs absolute",
              full[(full.loss == "absolute")], "wape")
    if r:
        rows.append(r)
    r = _show("TƯƠNG ĐỒNG PHÂN PHỐI (không phải khớp chính xác) — "
              "Tweedie (Single) vs Gamma (Two)",
              full[full.loss.isin(["tweedie", "gamma"])], "rmse")
    if r:
        rows.append(r)

    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "matched_loss_comparison.csv", index=False)
        print("\n  Hai bảng 'KHỚP CHÍNH XÁC' giữ cố định hàm mất mát, nên cô")
        print("  lập ảnh hưởng của kiến trúc chặt hơn — KHÔNG đồng nghĩa mọi")
        print("  chênh lệch đều do kiến trúc (xem docstring). Bảng 'TƯƠNG")
        print("  ĐỒNG PHÂN PHỐI' trả lời một câu hỏi khác và không nên gọi")
        print("  là 'matched loss' trong bài viết.")


def _recommend_model(results: pd.DataFrame, out_dir: Path,
                     gap: int, bias_tol: float = 0.05) -> str:
    """Chọn mô hình khuyến nghị dựa trên nhiều chỉ số, không chỉ một.

    Xếp hạng bằng một chỉ số duy nhất là không đủ trên bài toán này. Sai số
    tuyệt đối được tối thiểu hoá tại trung vị, mà trên dữ liệu nhiều số không
    trung vị nằm thấp hơn kỳ vọng, nên WAPE có xu hướng ưu ái những mô hình
    dự báo thấp một cách hệ thống. Một mô hình như vậy đứng đầu bảng WAPE
    nhưng gây hết hàng khi đưa vào vận hành.

    Quy tắc ở đây: trước hết loại những mô hình lệch quá ngưỡng cho phép, sau
    đó mới xếp hạng phần còn lại. Cách này ưu tiên tính dùng được trước, độ
    chính xác sau — đúng thứ tự ưu tiên của bài toán tồn kho.
    """
    _banner(f"MÔ HÌNH KHUYẾN NGHỊ (gap={gap})")
    print("  [Lưu ý] Đây là lựa chọn HẬU NGHIỆM dựa trên kết quả của chính "
          "tập kiểm tra\n  đang phân tích, không phải kết luận nhân quả và "
          "cũng không phải kết quả của\n  một quy trình chọn mô hình trên "
          "tập validation riêng. Khi báo cáo, nên trình\n  bày như 'cấu hình "
          "cho kết quả tốt nhất trong thực nghiệm này', không phải\n  'cấu "
          "hình được khuyến nghị cho triển khai thực tế'.\n")

    tab = (results[results.feature_set == "full"]
           .groupby("model", as_index=False)
           .agg(wape=("wape", "mean"), rmse=("rmse", "mean"),
                rmsse=("rmsse", "mean"), bias=("bias_ratio", "mean")))
    if not len(tab):
        return ""

    lo, hi = 1 - bias_tol, 1 + bias_tol
    tab["đạt_độ_chệch"] = tab.bias.between(lo, hi)

    print(f"  Điều kiện loại: độ chệch phải nằm trong [{lo:.2f}, {hi:.2f}]\n")
    print(tab.round(4).to_string(index=False))

    eligible = tab[tab["đạt_độ_chệch"]]
    if not len(eligible):
        print("\n  [CẢNH BÁO] không mô hình nào đạt ngưỡng độ chệch.")
        return ""

    # Xếp hạng trung bình trên ba chỉ số để không phụ thuộc vào một chỉ số
    ranks = eligible[["wape", "rmse", "rmsse"]].rank()
    eligible = eligible.assign(hạng_TB=ranks.mean(axis=1))
    eligible = eligible.sort_values("hạng_TB")

    best = eligible.iloc[0]
    print(f"\n  Sau khi loại theo độ chệch, xếp hạng trung bình ba chỉ số:")
    print(eligible[["model", "wape", "rmse", "rmsse", "bias", "hạng_TB"]]
          .round(4).to_string(index=False))
    print(f"\n  >>> KHUYẾN NGHỊ: {best.model}")
    print(f"      WAPE {best.wape:.4f} | RMSE {best.rmse:.4f} | "
          f"RMSSE {best.rmsse:.4f} | độ chệch {best.bias:.4f}")

    excluded = tab[~tab["đạt_độ_chệch"]].sort_values("wape")
    if len(excluded):
        top_wape = tab.nsmallest(1, "wape").iloc[0]
        if not top_wape["đạt_độ_chệch"]:
            print(f"\n  Lưu ý: mô hình dẫn đầu theo WAPE là {top_wape.model} "
                  f"(WAPE {top_wape.wape:.4f}),\n"
                  f"  nhưng bị loại vì độ chệch {top_wape.bias:.3f} — dự báo "
                  f"thiếu khoảng {(1-top_wape.bias)*100:.0f}%.\n"
                  f"  Đây chính là biểu hiện của việc WAPE thưởng cho dự báo "
                  f"thiếu.")

    eligible.to_csv(out_dir / "recommended_model.csv", index=False)
    return str(best.model)


def _seed_stability(results: pd.DataFrame, out_dir: Path, gap: int) -> None:
    """In bảng chi tiết từng seed, ưu tiên các cặp khớp chính xác hàm mất mát.

    Thứ tự ưu tiên ở đây PHẢI khớp với cách phân loại đã dùng ở
    ``_matched_loss_comparison`` và ``_run_significance``: cặp khớp chính xác
    (cùng hàm mất mát) cô lập ảnh hưởng của kiến trúc chặt hơn nên đặt lên đầu;
    cặp Tweedie/Gamma chỉ là tương đồng phân phối, đặt sau và gắn nhãn rõ để
    không bị hiểu nhầm là "câu hỏi trung tâm" của RQ1.

    Chỉ in khi kết quả có nhiều hơn 1 seed — với thực nghiệm chạy 1 seed, bảng
    này không có ý nghĩa và không nên xuất hiện.
    """
    if "seed" not in results.columns or results.seed.nunique() < 2:
        return

    _banner(f"ĐỘ ỔN ĐỊNH QUA SEED (gap={gap})")

    # "Single-Stage" mặc định dùng Tweedie, "Two-Stage" mặc định dùng
    # squared — hai nhãn ngắn này KHÔNG cùng hàm mất mát, nên không được ghép
    # với nhãn "cùng hàm mất mát squared" như bản trước đã làm sai. Cặp khớp
    # chính xác squared-squared phải dùng "Single-Stage[squared]".
    pairs = [
        ("Single-Stage[squared]", "Two-Stage",
         "Khớp chính xác — cùng hàm mất mát squared"),
        ("Single-Stage[absolute]", "Two-Stage[absolute]",
         "Khớp chính xác — cùng hàm mất mát absolute"),
        ("Single-Stage", "Two-Stage[gamma]",
         "Tương đồng phân phối (Tweedie≠Gamma) — không phải khớp chính xác"),
    ]

    rows_summary = []
    for model_a, model_b, label in pairs:
        tab = significance.seed_stability_table(results, model_a, model_b)
        if not len(tab):
            continue

        print(f"\n[{label}]")
        print(f"  {model_a}  vs  {model_b}")
        print(tab.round(4).to_string(index=False))

        s = significance.summarise_seed_stability(tab, model_a, model_b)
        win_col = f"{model_b}_thắng"
        print(f"\n  {model_b} thắng {s['n_wins_b']}/{s['n_seeds']} seed")
        print(f"  Δ trung bình     : {s['delta_mean']:+.4f}")
        print(f"  σ giữa các seed  : {s['delta_std']:.4f}")
        print(f"  |Δ| / σ          : {s['delta_over_std']:.1f}")
        print(f"  >>> Kết luận (dựa trên tỷ lệ {s['n_wins_b']}/{s['n_seeds']} "
              f"seed đồng thuận chiều thắng): {s['verdict']}")
        print(f"      (|Δ|/σ = {s['delta_over_std']:.1f} là chỉ báo ĐỘ LỚN "
              f"riêng biệt, không phải căn cứ của kết luận trên)")

        s.update({"model_a": model_a, "model_b": model_b, "label": label})
        rows_summary.append(s)

    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(
            out_dir / "seed_stability.csv", index=False)
        print(f"\nĐã lưu: {out_dir / 'seed_stability.csv'}")


def _assert_rq3_pair_matched(results: pd.DataFrame,
                             pair: tuple[str, str]) -> bool:
    """Xác minh cặp cố định của RQ3 thật sự dùng cùng một hàm mất mát.

    Kiểm tra bằng dữ liệu thay vì tin vào tên nhãn. Nhãn ngắn "Two-Stage" trỏ
    tới phần tử đầu tiên của ``stage2_objectives``, nên chỉ cần đổi thứ tự
    trong cấu hình là cặp hết khớp mà không có dấu hiệu nào báo lỗi — đúng
    dạng hỏng âm thầm mà toàn bộ phần kiểm định của repo này cố tránh.

    Trả về True nếu khớp. Nếu không khớp thì in cảnh báo và trả về False, để
    phần gọi quyết định có in bảng hay không — không tự ý im lặng bỏ qua.
    """
    if "loss" not in results.columns:
        return True                      # không đủ thông tin để kết luận

    a, b = pair
    losses = {}
    for m in pair:
        vals = results[results.model == m]["loss"].dropna().unique()
        if len(vals):
            losses[m] = vals[0]

    if len(losses) < 2:
        return True                      # thiếu mô hình, xử lý ở chỗ khác

    if losses[a] != losses[b]:
        print(f"\n  [CẢNH BÁO] Cặp cố định của RQ3 KHÔNG khớp hàm mất mát: "
              f"{a} dùng '{losses[a]}' còn {b} dùng '{losses[b]}'.")
        print("  Chênh lệch quan sát được sẽ lẫn giữa kiến trúc và hàm mất "
              "mát, nên\n  KHÔNG được diễn giải như một so sánh kiến trúc. "
              "Kiểm tra thứ tự\n  'stage2_objectives' và "
              "'single_stage_objectives' trong cấu hình.")
        return False
    return True


def _print_rq3_table(full: pd.DataFrame, pair: tuple[str, str],
                     title: str, out_path: Path) -> pd.DataFrame:
    """In và lưu bảng RQ3 cho một cặp (Single-Stage, Two-Stage) cho trước.

    Tách thành hàm dùng chung để Bảng A (cố định) và Bảng B (khám phá) có
    cùng định dạng, cùng logic tính cột chênh lệch — chỉ khác cách hai mô
    hình trong cặp được chọn ra từ trước.
    """
    a, b = pair
    print(f"\n[{title}]")
    print(f"  {a}  vs  {b}")

    pivot = (full.pivot_table(index="pattern", columns="model",
                              values="wape", aggfunc="mean").round(4))
    cols = [c for c in ["Croston", "SBA", "TSB", a, b] if c in pivot.columns]
    pivot = pivot[cols]

    if a in pivot.columns and b in pivot.columns:
        pivot["Two−Single"] = (pivot[b] - pivot[a]).round(4)
        pivot["Two thắng?"] = np.where(pivot["Two−Single"] < 0, "có", "không")

    ref_rows = full[full.model == b]
    info = (ref_rows.groupby("pattern")[["n_series", "zero_rate"]]
           .mean().round(4)) if len(ref_rows) else pd.DataFrame()
    out = info.join(pivot) if len(info) else pivot
    order_p = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    out = out.reindex([p for p in order_p if p in out.index])

    print(out.to_string())
    print("  Cột 'Two−Single' âm nghĩa là khung hai giai đoạn tốt hơn ở "
          "nhóm đó.")
    out.to_csv(out_path)
    return out


def _summarise(results: pd.DataFrame, strat: pd.DataFrame | None,
               out_dir: Path, gap: int) -> None:
    """In các bảng tổng hợp tương ứng từng câu hỏi nghiên cứu."""
    # RMSSE loại bỏ những chuỗi không có hệ số chuẩn hoá hợp lệ (chuỗi huấn
    # luyện không đủ biến thiên). ``metrics.py`` đã đếm sẵn nhưng trước đây
    # không đưa ra bảng tổng hợp, nên người đọc không biết RMSSE được tính
    # trên bao nhiêu chuỗi. Báo cáo tường minh: nếu số bị loại bằng 0 thì
    # nhìn vào là biết ngay RMSSE không mất mẫu.
    agg_spec = dict(
        wape=("wape", "mean"), mae=("mae", "mean"),
        rmse=("rmse", "mean"), rmsse=("rmsse", "mean"),
        bias=("bias_ratio", "mean"),
        near_zero=("near_zero_rate", "mean"),
        seconds=("seconds", "mean"))
    for col in ("rmsse_n_series", "rmsse_n_excluded"):
        if col in results.columns:
            agg_spec[col] = (col, "mean")

    agg = (results.groupby(["model", "feature_set"], as_index=False)
           .agg(**agg_spec)
           .sort_values("wape"))

    if "rmsse_n_series" in agg.columns and "rmsse_n_excluded" in agg.columns:
        total = agg.rmsse_n_series + agg.rmsse_n_excluded
        agg["rmsse_exclusion_rate"] = (
            agg.rmsse_n_excluded / total.replace(0, np.nan)).round(4)

    agg.to_csv(out_dir / "summary.csv", index=False)

    _banner(f"RQ1 — SO SÁNH MÔ HÌNH (gap={gap}, trung bình các fold)")
    # Giữ bảng chính gọn để dễ đọc; số liệu về độ phủ RMSSE in riêng bên dưới
    # và luôn có đủ trong summary.csv.
    audit_cols = {"rmsse_n_series", "rmsse_n_excluded", "rmsse_exclusion_rate"}
    main_cols = [c for c in agg.columns if c not in audit_cols]
    print(agg[main_cols].round(4).to_string(index=False))

    if "rmsse_n_excluded" in agg.columns:
        n_excl = float(agg.rmsse_n_excluded.mean())
        n_kept = float(agg.rmsse_n_series.mean())
        if n_excl <= 0:
            print(f"\n  RMSSE tính trên toàn bộ {n_kept:,.0f} chuỗi — không "
                  f"chuỗi nào bị loại.")
        else:
            rate = n_excl / (n_excl + n_kept) if (n_excl + n_kept) else float("nan")
            print(f"\n  [LƯU Ý] RMSSE tính trên {n_kept:,.0f} chuỗi; "
                  f"{n_excl:,.0f} chuỗi bị loại ({rate:.2%}) do chuỗi huấn "
                  f"luyện\n  không đủ biến thiên để có hệ số chuẩn hoá hợp "
                  f"lệ. Con số này cần nêu khi\n  báo cáo RMSSE — xem cột "
                  f"rmsse_n_* trong summary.csv.")

    _seed_stability(results, out_dir, gap)
    _bias_diagnostics(results, gap)
    _matched_loss_comparison(results, out_dir, gap)
    recommended = _recommend_model(results, out_dir, gap)

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

    # Hướng dẫn đọc bảng chỉ đúng khi có nhiều bộ đặc trưng xếp chồng. Khi
    # lượt chạy chỉ có một bộ (ví dụ Phase A chỉ chạy 'full'), in hướng dẫn
    # về "đóng góp riêng của từng nhóm" là mô tả một thứ không tồn tại trong
    # kết quả — dễ khiến người đọc tưởng RQ2 đã được trả lời.
    if len(order) > 1:
        print("\n  Đọc bảng: mỗi dòng thêm đúng một nhóm đặc trưng so với "
              "dòng trên,\n  nên cột 'riêng_%' là đóng góp của riêng nhóm đó.")
        if "hist_cal_hol_promo" in order:
            print("  Dòng 'hist_cal_hol_promo' cô lập đóng góp của KHUYẾN MÃI.")
    else:
        only = order[0] if order else "(không có)"
        print(f"\n  [RQ2 CHƯA ĐƯỢC TRẢ LỜI] lượt chạy này chỉ có một bộ đặc "
              f"trưng: '{only}'.")
        print("  Không có nghiên cứu loại trừ nên KHÔNG thể kết luận gì về "
              "đóng góp của\n  từng nhóm đặc trưng. Muốn trả lời RQ2 phải "
              "chạy nhiều bộ trong\n  'experiment.feature_sets' (Phase B).")

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
    print("  [Lưu ý] Đây là phân tích phân nhóm hồi cứu: nhãn Smooth/Erratic/"
          "Intermittent/Lumpy\n  được tính trên toàn bộ cửa sổ nghiên cứu, "
          "không phải thông tin có sẵn tại\n  thời điểm dự báo thực tế của "
          "từng fold. Không dùng làm đặc trưng đầu vào.")
    full = strat[strat.feature_set.isin(["full", "-"])]

    # ---- BẢNG A (CHÍNH): cặp CỐ ĐỊNH, KHỚP hàm mất mát ----
    # Cặp được cố định trước khi xem kết quả test VÀ dùng chung một hàm mất
    # mát (xem chú thích ở ``RQ3_FIXED_PAIR``). Hai điều kiện này phải có đủ:
    # cố định mà không khớp hàm mất mát thì vẫn không tách được ảnh hưởng của
    # kiến trúc khỏi ảnh hưởng của hàm mất mát.
    fixed_pair = RQ3_FIXED_PAIR
    missing = [m for m in fixed_pair if m not in set(full.model)]
    if missing:
        print(f"\n  [BỎ QUA BẢNG A] thiếu mô hình {missing} trong kết quả.")
        print("  Bảng A cần cặp khớp hàm mất mát; hãy bật 'squared' trong cả")
        print("  'single_stage_objectives' lẫn 'stage2_objectives' rồi chạy "
              "lại.\n  KHÔNG dùng Bảng B thay thế — đó là phân tích khám phá.")
    else:
        _assert_rq3_pair_matched(results, fixed_pair)
        _print_rq3_table(
            full, fixed_pair,
            title="BẢNG A (CHÍNH, xác nhận) — cặp cố định, KHỚP hàm mất mát "
                  "(squared vs squared), không phụ thuộc kết quả test",
            out_path=out_dir / "rq3_fixed_pair_by_pattern.csv")

    # ---- BẢNG B (PHỤ, khám phá): biến thể tốt nhất chọn SAU khi xem test ----
    # Cảnh báo phương pháp: biến thể "tốt nhất" của mỗi kiến trúc được chọn
    # bằng WAPE/độ chệch tính trên CHÍNH tập kiểm tra dùng để phân tích dưới
    # đây. Đây là một dạng "nhìn kết quả rồi chọn nhà vô địch, rồi phân tích
    # tiếp trên chính kết quả đó" — không phải leakage (không mô hình nào
    # huấn luyện trên dữ liệu kiểm tra), nhưng khiến bảng này mang tính khám
    # phá (exploratory) chứ không phải xác nhận (confirmatory). Không dùng
    # bảng này làm bằng chứng chính cho RQ3; chỉ dùng Bảng A cho việc đó.
    best = {}
    if "arch" in full.columns and "bias_ratio" in full.columns:
        for arch in ("Single-Stage", "Two-Stage"):
            sub = full[full.arch == arch]
            if not len(sub):
                continue
            agg = sub.groupby("model").agg(
                wape=("wape", "mean"), bias=("bias_ratio", "mean"))
            ok = agg[agg.bias.between(0.95, 1.05)]
            best[arch] = (ok if len(ok) else agg).wape.idxmin()

    _summarise.best_pair = (best.get("Single-Stage"), best.get("Two-Stage"))

    if len(best) == 2:
        print("\n  Biến thể tốt nhất của mỗi kiến trúc theo test set "
              "(đã loại biến thể lệch quá 5%):")
        for k, v in best.items():
            b = full[full.model == v]["bias_ratio"].mean()
            print(f"    {k:<14} -> {v:<26} (độ chệch {b:.3f})")
        _print_rq3_table(
            full, (best["Single-Stage"], best["Two-Stage"]),
            title="BẢNG B (PHỤ, khám phá — KHÔNG dùng làm bằng chứng chính) "
                  "— biến thể chọn sau khi xem test",
            out_path=out_dir / "rq3_exploratory_best_by_pattern.csv")

    print(f"\nKết quả lưu tại: {out_dir}")


def _rq3_significance(losses: pd.DataFrame, series: pd.DataFrame,
                      pair: tuple[str, str], out_dir: Path,
                      title: str, filename: str,
                      caveat: str | None = None) -> None:
    """Kiểm định RQ3 theo từng nhóm nhu cầu cho một cặp mô hình cho trước.

    Tách thành hàm dùng chung để cặp cố định (xác nhận) và cặp chọn sau khi
    xem test (khám phá) có cùng định dạng, chỉ khác nhãn và cảnh báo kèm
    theo — giúp người đọc so sánh trực tiếp hai kết quả và tự thấy chênh
    lệch giữa chúng, thay vì chỉ được đưa một con số duy nhất.
    """
    a_model, b_model = pair
    a_key, b_key = f"{a_model}|full", f"{b_model}|full"
    keys = set(losses.model_key)
    if a_key not in keys or b_key not in keys:
        return

    print("\n" + "-" * 72)
    print(f"  {title}")
    print(f"  {a_key}  vs  {b_key}")
    if caveat:
        print(f"  [CẢNH BÁO] {caveat}")
    print("-" * 72)

    by_pat = significance.compare_by_pattern(losses, series, a_key, b_key)
    if not len(by_pat):
        return

    show = by_pat[["pattern", "n_series", "mean_diff", "b_win_rate",
                   "p_wilcoxon", "cohen_d", "độ_lớn", "ý_nghĩa"]].rename(
        columns={"mean_diff": "chênh_MAE", "b_win_rate": "tỷ_lệ_thắng"})
    show = show.round(5)
    show["p_wilcoxon"] = by_pat.p_wilcoxon.map(_fmt_p)
    print(show.to_string(index=False))
    print("\n  chênh_MAE dương nghĩa là khung hai giai đoạn tốt hơn "
          "trong nhóm đó.")
    print("  tỷ_lệ_thắng là tỷ lệ chuỗi mà khung hai giai đoạn thắng.")
    print("  Cột ý_nghĩa đã hiệu chỉnh Holm cho bốn nhóm.")

    # Tổng kết phải phân biệt BA tình huống, không phải hai.
    #
    # Bản trước chỉ đếm số nhóm "thắng có ý nghĩa" rồi in "thắng ở 1/4 nhóm".
    # Câu đó khiến người đọc hiểu ba nhóm còn lại không có khác biệt, trong
    # khi thực tế chúng có thể khác biệt có ý nghĩa theo chiều NGƯỢC LẠI —
    # tức khung hai giai đoạn THUA. Gộp "thua có ý nghĩa" chung với "không
    # khác biệt" là một sai sót diễn giải, không phải sai sót tính toán.
    n = len(by_pat)
    won = by_pat[(by_pat.mean_diff > 0) & by_pat.reject_holm]
    lost = by_pat[(by_pat.mean_diff < 0) & by_pat.reject_holm]
    tie = by_pat[~by_pat.reject_holm]

    def _names(sub: pd.DataFrame) -> str:
        return f": {', '.join(sub.pattern)}" if len(sub) else ""

    print(f"\n  Phân định theo chiều và mức ý nghĩa (tổng {n} nhóm):")
    print(f"    Hai giai đoạn TỐT HƠN có ý nghĩa : "
          f"{len(won)}/{n}{_names(won)}")
    print(f"    Hai giai đoạn KÉM HƠN có ý nghĩa : "
          f"{len(lost)}/{n}{_names(lost)}")
    print(f"    Không đủ bằng chứng khác biệt    : "
          f"{len(tie)}/{n}{_names(tie)}")
    print("\n  Lưu ý khi viết bài: 'không thắng' KHÁC 'không có khác biệt'. "
          "Với cỡ mẫu\n  hàng chục nghìn chuỗi, phần lớn nhóm sẽ đạt ý nghĩa "
          "thống kê; cột độ_lớn\n  mới cho biết khác biệt có đáng kể trong "
          "thực tế hay không.")
    by_pat.to_csv(out_dir / filename, index=False)


def _run_significance(losses: pd.DataFrame, results: pd.DataFrame,
                      out_dir: Path, gap: int,
                      series: pd.DataFrame | None = None,
                      best_pair: tuple | None = None) -> None:
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
    print(f"  Mô hình tham chiếu: {reference}")
    print("  [CẢNH BÁO] Mô hình tham chiếu này được chọn bằng WAPE tính trên")
    print("  CHÍNH tập kiểm tra đang phân tích, không phải trên tập validation")
    print("  riêng. Việc so mọi mô hình khác với 'người thắng' xác định từ")
    print("  cùng dữ liệu khiến giá trị p ở bảng dưới lạc quan hơn thực tế.")
    print("  Bảng này dùng để XẾP HẠNG và mô tả, không dùng làm bằng chứng")
    print("  xác nhận cho giả thuyết nào. Bằng chứng xác nhận cho RQ1 nằm ở")
    print("  các cặp khớp chính xác hàm mất mát bên dưới.\n")

    table = significance.compare_all(losses, reference)
    if not len(table):
        print("  Không đủ dữ liệu để kiểm định")
        return

    show = table[["model_b", "n_series", "mean_diff", "b_win_rate",
                  "p_wilcoxon", "cohen_d", "độ_lớn", "ý_nghĩa"]].rename(
        columns={"model_b": "so_với", "mean_diff": "chênh_MAE",
                 "b_win_rate": "tỷ_lệ_thắng"})
    show = show.round(5)
    show["p_wilcoxon"] = table.p_wilcoxon.map(_fmt_p)
    print(show.to_string(index=False))
    print("\n  chênh_MAE âm nghĩa là mô hình tham chiếu tốt hơn.")
    print("  tỷ_lệ_thắng là tỷ lệ chuỗi mà mô hình kia thắng tham chiếu.")
    print("  Cột ý_nghĩa đã hiệu chỉnh Holm cho nhiều phép so sánh.")
    print("\n  Với hai mươi lăm nghìn chuỗi, lực kiểm định cao tới mức gần "
          "như mọi chênh\n  lệch đều đạt ý nghĩa thống kê. Cột độ_lớn mới cho "
          "biết khác biệt có đáng\n  kể trong thực tế hay không.")
    print("\n  Lưu ý về đơn vị phân tích: các chuỗi không hoàn toàn độc lập —"
          " nhiều chuỗi\n  chung cửa hàng hoặc cùng giai đoạn thời gian có "
          "thể tương quan với nhau.\n  Giá trị p ở đây nên được đọc như một "
          "chỉ báo tương đối, không phải bằng\n  chứng cho N quan sát độc lập"
          " hoàn toàn. Độ lớn hiệu ứng và tỷ lệ thắng là\n  các chỉ số đáng "
          "tin cậy hơn để đánh giá mức độ quan trọng thực tế.")
    table.to_csv(out_dir / "significance_vs_best.csv", index=False)

    # So sánh trực tiếp cặp quan trọng nhất của RQ1
    pooled = (losses.groupby(["model_key", "store_nbr", "item_nbr"],
                             as_index=False).agg(mae=("mae", "mean")))
    # Ba nhóm so sánh, KHÔNG gộp chung một nhãn — mỗi nhóm trả lời một câu
    # hỏi khác nhau và có độ chặt chẽ khác nhau.
    exact_match_pairs = [
        ("Single-Stage[squared]|full", "Two-Stage|full"),
        ("Single-Stage[absolute]|full", "Two-Stage[absolute]|full"),
    ]
    analogy_pairs = [
        ("Single-Stage|full", "Two-Stage[gamma]|full"),
    ]
    # Cặp mặc định của mỗi kiến trúc (Tweedie vs squared) — hai hàm mất mát
    # khác nhau, không thuộc nhóm nào ở trên. Giữ lại vì đây là điểm khởi đầu
    # đã dẫn tới toàn bộ điều tra về hàm mất mát, nhưng không được gọi là
    # "khớp" dưới bất kỳ hình thức nào.
    unmatched_reference_pairs = [
        ("Single-Stage|full", "Two-Stage|full"),
    ]

    def _print_group(label: str, pairs: list) -> list:
        rows = [significance.paired_test_by_series(pooled, a, b)
                for a, b in pairs
                if a in set(pooled.model_key) and b in set(pooled.model_key)]
        if rows:
            print(f"\n  [{label}]")
            for r in rows:
                print(f"    {r['model_a']}\n      vs {r['model_b']}")
                print(f"      chênh MAE trung bình : {r['mean_diff']:+.5f}")
                print(f"      tỷ lệ chuỗi B thắng  : {r['b_win_rate']:.1%}")
                print(f"      p (Wilcoxon)         : "
                      f"{_fmt_p(r['p_wilcoxon'])}")
                print(f"      độ lớn hiệu ứng      : {r['cohen_d']:.4f} "
                      f"({r['độ_lớn']})")
                print(f"      kết luận             : {r['verdict']}")
        return rows

    print("\n  RQ1 — so sánh ghép cặp, tách theo mức độ khớp hàm mất mát:")
    all_rows = []
    for r in _print_group(
            "KHỚP CHÍNH XÁC — hàm mất mát cố định, cô lập kiến trúc chặt hơn",
            exact_match_pairs):
        r["category"] = "exact_match"
        all_rows.append(r)
    for r in _print_group(
            "TƯƠNG ĐỒNG PHÂN PHỐI — Tweedie≠Gamma, không phải khớp chính xác",
            analogy_pairs):
        r["category"] = "distributional_analogy"
        all_rows.append(r)
    for r in _print_group(
            "MỐC THAM CHIẾU CHƯA KHỚP — Tweedie vs squared, để đối chiếu",
            unmatched_reference_pairs):
        r["category"] = "unmatched_reference"
        all_rows.append(r)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            out_dir / "significance_rq1.csv", index=False)

    # ---- Kiểm định riêng trong từng nhóm mẫu nhu cầu (RQ3) ----
    #
    # Chạy HAI lần, tách bạch đúng như Bảng A/Bảng B ở phần mô tả:
    #
    #   Cặp cố định (chính, xác nhận) — Single-Stage vs Two-Stage mặc định,
    #       khai báo trước khi xem kết quả test. Đây là cặp duy nhất mà
    #       p-value và độ lớn hiệu ứng diễn giải được theo nghĩa suy luận
    #       thống kê thông thường.
    #
    #   Cặp chọn sau khi xem test (phụ, khám phá) — biến thể "tốt nhất" của
    #       mỗi kiến trúc được chọn bằng WAPE/độ chệch tính trên CHÍNH tập
    #       kiểm tra này. Kiểm định trên cặp đó là kiểm định trên giả thuyết
    #       sinh ra từ dữ liệu, nên p-value bị lạc quan và KHÔNG được trình
    #       bày như bằng chứng xác nhận.
    if series is not None:
        # Dùng chung hằng số với phần mô tả để hai mục không thể lệch nhau.
        fixed_pair = RQ3_FIXED_PAIR
        _rq3_significance(
            losses, series, fixed_pair, out_dir,
            title="RQ3 (CHÍNH, xác nhận) — cặp cố định, KHỚP hàm mất mát "
                  "(squared vs squared)",
            filename="significance_rq3_fixed_pair.csv",
            caveat=None)

        if best_pair and all(best_pair) and tuple(best_pair) != fixed_pair:
            _rq3_significance(
                losses, series, tuple(best_pair), out_dir,
                title="RQ3 (PHỤ, khám phá) — cặp chọn SAU khi xem test",
                filename="significance_rq3_exploratory.csv",
                caveat="Cặp này được chọn dựa trên kết quả của chính tập "
                       "kiểm tra đang phân tích.\n      Giá trị p ở đây "
                       "lạc quan hơn thực tế và KHÔNG dùng làm bằng chứng "
                       "xác nhận.")

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
    ap.add_argument("--summarize-only", action="store_true",
                    help="dựng lại toàn bộ bảng tổng hợp từ checkpoint đã có, "
                         "không huấn luyện lại mô hình")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gaps = ([args.gap] if args.gap is not None
            else cfg.split.get("gap_variants", [cfg.split["gap_days"]]))

    frames = []
    for gap in gaps:
        frames.append(run_experiment(cfg, gap, args.max_parts, args.quick,
                                     args.summarize_only))

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
