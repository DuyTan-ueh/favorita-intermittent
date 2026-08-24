"""Kiểm thử chống rò rỉ thông tin và tính toàn vẹn của lưới.

Chạy: ``pytest tests/ -v``

Bộ kiểm thử này quan trọng hơn vẻ ngoài của nó. Rò rỉ thông tin tương lai
không làm chương trình báo lỗi — nó chỉ làm kết quả đẹp lên một cách giả tạo,
và thường chỉ bị phát hiện khi phản biện đặt câu hỏi. Tự động hoá việc kiểm
tra là cách duy nhất để yên tâm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checks import LeakageError, rolling_origin_folds  # noqa: E402
from src.config import Config, ConfigError, _validate  # noqa: E402
from src.data import classify_pattern  # noqa: E402
from src.features import (Availability, add_intermittency_features,  # noqa: E402
                          add_lag_features, add_rolling_features,
                          build_specs, validate_specs)
from src.grid import GridIntegrityError, assert_grid_complete  # noqa: E402


# --------------------------------------------------------------------------- #
# Đồ gá kiểm thử
# --------------------------------------------------------------------------- #
def make_cfg(horizon: int = 7, lags=None, windows=None) -> Config:
    raw = {
        "run": {"name": "test", "seed": 42},
        "data": {"start_date": "2014-04-01", "end_date": None,
                 "min_days_active": 365, "min_positive_days": 50,
                 "require_active_until_end": 60},
        "sampling": {"enabled": False, "n_series": 1000,
                     "stratify_by": "pattern"},
        "forecast": {"horizon": horizon},
        "features": {"lags": lags or [7, 14], "rolling_windows": windows or [7],
                     "rolling_stats": ["mean"], "use_calendar": True,
                     "use_promotion": True, "use_holidays": False,
                     "use_static": False},
        "split": {"n_folds": 3, "test_days": 28, "gap_days": 0},
        "output": {"dir": "artifacts", "format": "parquet", "batch_size": 100},
    }
    return Config(raw=raw, raw_dir=Path("."), out_dir=Path("."),
                  **{k: raw[k] for k in
                     ["run", "data", "sampling", "forecast", "features",
                      "split", "output"]})


def make_grid(n_series: int = 3, n_days: int = 120, seed: int = 0):
    """Lưới giả lập với nhu cầu gián đoạn, đầy đủ từng ngày."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_series):
        dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
        y = np.where(rng.random(n_days) < 0.35, rng.poisson(4, n_days) + 1, 0)
        frames.append(pd.DataFrame({
            "date": dates,
            "store_nbr": np.int16(1),
            "item_nbr": np.int32(100 + s),
            "y": y.astype("float32"),
            "onpromotion": rng.integers(0, 2, n_days).astype("int8"),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
class TestConfigGuards:
    def test_lag_smaller_than_horizon_is_rejected(self):
        """Lag < horizon là rò rỉ — phải bị chặn ngay từ cấu hình."""
        raw = make_cfg().raw
        raw["forecast"]["horizon"] = 7
        raw["features"]["lags"] = [1, 7]        # lag_1 không hợp lệ
        with pytest.raises(ConfigError, match="LEAKAGE"):
            _validate(raw)

    def test_valid_lags_pass(self):
        raw = make_cfg().raw
        raw["forecast"]["horizon"] = 7
        raw["features"]["lags"] = [7, 14, 28]
        _validate(raw)                           # không ném lỗi

    def test_unknown_rolling_stat_rejected(self):
        raw = make_cfg().raw
        raw["features"]["rolling_stats"] = ["mean", "kurtosis"]
        with pytest.raises(ConfigError):
            _validate(raw)


# --------------------------------------------------------------------------- #
# Khai báo đặc trưng
# --------------------------------------------------------------------------- #
class TestFeatureSpecs:
    def test_all_lagged_specs_respect_horizon(self):
        cfg = make_cfg(horizon=7, lags=[7, 14, 28])
        specs = build_specs(cfg)
        validate_specs(cfg, specs)
        for s in specs:
            if s.availability is Availability.LAGGED:
                assert s.min_lag >= cfg.horizon

    def test_promotion_declared_known_in_advance(self):
        """Khuyến mãi là kế hoạch công bố trước nên hợp lệ ở ngày đích."""
        cfg = make_cfg()
        specs = {s.name: s for s in build_specs(cfg)}
        assert specs["onpromotion"].availability is Availability.KNOWN_IN_ADVANCE

    def test_no_duplicate_feature_names(self):
        cfg = make_cfg(windows=[7, 14, 28])
        specs = build_specs(cfg)
        validate_specs(cfg, specs)


# --------------------------------------------------------------------------- #
# Toàn vẹn lưới
# --------------------------------------------------------------------------- #
class TestGridIntegrity:
    def test_complete_grid_passes(self):
        assert_grid_complete(make_grid())

    def test_missing_day_is_detected(self):
        """Lưới thủng khiến shift lấy sai ngày — phải bị bắt."""
        g = make_grid(n_series=1).drop(index=[10, 11, 12]).reset_index(drop=True)
        with pytest.raises(GridIntegrityError):
            assert_grid_complete(g)

    def test_duplicate_rows_detected(self):
        g = make_grid(n_series=1)
        g = pd.concat([g, g.iloc[[5]]], ignore_index=True)
        g = g.sort_values(["store_nbr", "item_nbr", "date"])
        with pytest.raises(GridIntegrityError):
            assert_grid_complete(g)


# --------------------------------------------------------------------------- #
# Tính đúng đắn của đặc trưng
# --------------------------------------------------------------------------- #
class TestFeatureCorrectness:
    def test_lag_matches_manual_shift(self):
        cfg = make_cfg(horizon=7, lags=[7])
        g = make_grid(n_series=1, n_days=60)
        out = add_lag_features(g.copy(), cfg)
        expected = g.y.shift(7)
        pd.testing.assert_series_equal(
            out.lag_7.reset_index(drop=True).astype("float64"),
            expected.reset_index(drop=True).astype("float64"),
            check_names=False)

    def test_rolling_excludes_current_row(self):
        """Cửa sổ trượt không được chứa giá trị của chính ngày đích."""
        cfg = make_cfg(horizon=7, windows=[7])
        g = make_grid(n_series=1, n_days=60)
        out = add_rolling_features(g.copy(), cfg)

        # với horizon=7, giá trị tại dòng i phải tính từ y[i-13 .. i-7]
        i = 30
        window = g.y.iloc[i - 13:i - 6]
        assert abs(out.roll_mean_7.iloc[i] - window.mean()) < 1e-5

    def test_days_since_last_demand_is_lagged(self):
        cfg = make_cfg(horizon=7)
        g = make_grid(n_series=1, n_days=60)
        out = add_intermittency_features(g.copy(), cfg)
        assert out.days_since_last_demand.notna().any()
        # không được dùng giá trị của chính ngày đích
        assert out.days_since_last_demand.iloc[:7].isna().all()


# --------------------------------------------------------------------------- #
# Kiểm định bất biến cắt cụt — phép thử tổng quát nhất
# --------------------------------------------------------------------------- #
class TestTruncationInvariance:
    @staticmethod
    def _engineer(df, cfg):
        df = add_lag_features(df, cfg)
        df = add_rolling_features(df, cfg)
        df = add_intermittency_features(df, cfg)
        return df

    def test_features_unchanged_when_future_removed(self):
        """Xoá dữ liệu tương lai không được đổi đặc trưng của quá khứ."""
        from src.checks import truncation_invariance_test
        cfg = make_cfg(horizon=7, lags=[7, 14], windows=[7, 14])
        g = make_grid(n_series=2, n_days=120)
        cut = g.date.max() - pd.Timedelta(days=40)
        bad = truncation_invariance_test(self._engineer, g, cfg, cut)
        assert len(bad) == 0, f"Đặc trưng rò rỉ: {bad.to_dict('records')}"

    def test_detector_catches_deliberate_leak(self):
        """Cố tình tạo rò rỉ để chứng minh phép thử thực sự phát hiện được."""
        from src.checks import truncation_invariance_test
        cfg = make_cfg(horizon=7, lags=[7], windows=[7])

        def leaky(df, c):
            df = add_lag_features(df, c)
            # trung bình toàn chuỗi: dùng cả dữ liệu tương lai
            df["leak"] = df.groupby(["store_nbr", "item_nbr"])["y"].transform("mean")
            return df

        g = make_grid(n_series=2, n_days=120)
        cut = g.date.max() - pd.Timedelta(days=40)
        bad = truncation_invariance_test(leaky, g, cfg, cut)
        assert "leak" in set(bad.feature), "Phép thử KHÔNG bắt được rò rỉ cố ý"


# --------------------------------------------------------------------------- #
# Chia tập theo thời gian
# --------------------------------------------------------------------------- #
class TestSplits:
    def test_folds_are_chronological(self):
        cfg = make_cfg()
        df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=400)})
        for f in rolling_origin_folds(df, cfg):
            assert f["train_end"] < f["test_start"]

    def test_test_windows_do_not_overlap(self):
        cfg = make_cfg()
        df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=400)})
        folds = rolling_origin_folds(df, cfg)
        for a, b in zip(folds, folds[1:]):
            assert a["test_end"] < b["test_start"]


# --------------------------------------------------------------------------- #
# Phân loại mẫu nhu cầu
# --------------------------------------------------------------------------- #
class TestPatternClassification:
    def test_four_quadrants(self):
        adi = pd.Series([1.0, 1.0, 5.0, 5.0])
        cv2 = pd.Series([0.2, 1.0, 0.2, 1.0])
        got = classify_pattern(adi, cv2).tolist()
        assert got == ["Smooth", "Erratic", "Intermittent", "Lumpy"]

    def test_nan_is_unclassified(self):
        adi = pd.Series([np.nan])
        cv2 = pd.Series([0.5])
        assert classify_pattern(adi, cv2).iloc[0] == "Unclassified"
