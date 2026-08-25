"""Điều phối toàn bộ pipeline: dữ liệu thô -> lưới -> đặc trưng -> tệp kết quả.

Chạy từ dòng lệnh::

    python -m src.pipeline --config config/default.yaml
    python -m src.pipeline --config config/default.yaml --stage stats
    python -m src.pipeline --smoke        # chạy nhanh trên tập nhỏ để kiểm thử
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import checks, data, features, grid
from .config import Config, load_config


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def run_stats(cfg: Config) -> pd.DataFrame:
    """Bước 1 — thống kê mô tả và chọn tập chuỗi."""
    _banner("BƯỚC 1 — THỐNG KÊ CHUỖI VÀ CHỌN MẪU")
    path = cfg.out_dir / "series_stats.parquet"

    if path.exists():
        print(f"Dùng lại kết quả đã có: {path}")
        stats = pd.read_parquet(path)
    else:
        t0 = time.time()
        stats = data.compute_series_stats(cfg)
        stats.to_parquet(path, index=False)
        print(f"Đã tính {len(stats):,} chuỗi trong {time.time() - t0:.0f}s")

    selected = data.select_series(cfg, stats)
    selected.to_parquet(cfg.out_dir / "series_selected.parquet", index=False)
    return selected


def run_features(cfg: Config, selected: pd.DataFrame) -> None:
    """Bước 2 — dựng lưới, sinh đặc trưng, ghi ra tệp theo lô."""
    _banner("BƯỚC 2 — DỰNG LƯỚI VÀ SINH ĐẶC TRƯNG")

    specs = features.build_specs(cfg)
    features.validate_specs(cfg, specs)
    summary = features.summarise_specs(specs)
    summary.to_csv(cfg.out_dir / "feature_specs.csv", index=False)

    print(f"\n{len(specs)} đặc trưng, phân theo tính khả dụng:")
    print(summary.availability.value_counts().to_string())
    print(f"\nHorizon = {cfg.horizon} ngày | lag nhỏ nhất = {min(cfg.lags)}")
    print("  [đạt] mọi đặc trưng trễ đều >= horizon")

    items = data.load_items(cfg)
    stores = data.load_stores(cfg)
    holidays = data.load_holidays(cfg)

    sales = grid.load_filtered_sales(cfg, selected)

    def _engineer(df: pd.DataFrame, c: Config) -> pd.DataFrame:
        return features.engineer(df, c, items, stores, holidays)

    feat_dir = cfg.out_dir / "features"
    feat_dir.mkdir(exist_ok=True)

    n_rows = 0
    leak_checked = False
    batch_stats = []

    for i, g in grid.iter_grid_batches(cfg, selected, sales):
        out = _engineer(g, cfg)

        # Kiểm định rò rỉ một lần trên lô đầu — đủ để bắt lỗi cài đặt,
        # chạy mọi lô sẽ tốn thời gian mà không thêm thông tin
        if not leak_checked:
            _banner("KIỂM ĐỊNH RÒ RỈ THÔNG TIN")
            sample_keys = g[["store_nbr", "item_nbr"]].drop_duplicates().head(50)
            sub = g.merge(sample_keys, on=["store_nbr", "item_nbr"])
            cut = sub.date.max() - pd.Timedelta(days=60)
            checks.assert_no_leakage(_engineer, sub, cfg, cut)
            checks.check_feature_nulls(out, cfg)
            leak_checked = True

        batch_stats.append({
            "batch": i,
            "n_rows": len(out),
            "n_series": out.groupby(["store_nbr", "item_nbr"]).ngroups,
            "zero_pct": round(float((out.y == 0).mean() * 100), 1),
        })

        out.to_parquet(feat_dir / f"part_{i:04d}.parquet", index=False)
        n_rows += len(out)
        del out, g
        gc.collect()

    homogeneity = checks.check_batch_homogeneity(batch_stats)
    homogeneity.to_csv(cfg.out_dir / "batch_stats.csv", index=False)

    _banner("ĐỊNH NGHĨA FOLD")
    calendar = pd.DataFrame({"date": pd.date_range(
        selected.first_date.min(), selected.global_end.iloc[0])})
    fold_variants = checks.build_fold_variants(calendar, cfg)

    for gap, folds in fold_variants.items():
        path = cfg.out_dir / f"folds_gap{gap}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([{k: str(v) for k, v in f.items()} for f in folds],
                      fh, indent=2)

    meta = {
        "run_name": cfg.run["name"],
        "n_series": int(len(selected)),
        "n_rows": int(n_rows),
        "horizon": cfg.horizon,
        "n_features": len(specs),
        "gap_variants": list(fold_variants.keys()),
        "folds": {str(gap): [{k: str(v) for k, v in f.items()} for f in folds]
                  for gap, folds in fold_variants.items()},
        "config": cfg.raw,
    }
    with open(cfg.out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    _banner("HOÀN TẤT")
    print(f"  {n_rows:,} dòng | {len(specs)} đặc trưng")
    print(f"  Bộ fold: {', '.join(f'gap={g}' for g in fold_variants)}")
    print(f"  Kết quả: {cfg.out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline dữ liệu Favorita")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--stage", choices=["all", "stats", "features"],
                    default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="chạy nhanh trên tập rất nhỏ để kiểm thử pipeline")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.smoke:
        cfg.sampling["enabled"] = True
        cfg.sampling["n_series"] = 200
        cfg.output["batch_size"] = 100
        cfg.run["name"] += "_smoke"
        cfg.out_dir = Path(cfg.output["dir"]) / cfg.run["name"]
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        print(">>> CHẾ ĐỘ SMOKE TEST: 200 chuỗi <<<")

    print(f"Cấu hình : {args.config}")
    print(f"Dữ liệu  : {cfg.raw_dir}")
    print(f"Kết quả  : {cfg.out_dir}")

    selected = run_stats(cfg)
    if args.stage in ("all", "features"):
        run_features(cfg, selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
