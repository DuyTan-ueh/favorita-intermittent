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
        "split": {"n_folds": 3, "test_days": 28, "gap_days": 0,
                  "gap_variants": [0, 7]},
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

    def test_gap_shrinks_training_window(self):
        """Tăng gap phải làm tập huấn luyện kết thúc sớm hơn."""
        cfg = make_cfg()
        df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=400)})
        f0 = rolling_origin_folds(df, cfg, gap_days=0, verbose=False)
        f7 = rolling_origin_folds(df, cfg, gap_days=7, verbose=False)
        for a, b in zip(f0, f7):
            assert b["train_end"] < a["train_end"]
            assert (a["train_end"] - b["train_end"]).days == 7
            # cửa sổ kiểm tra phải giữ nguyên để hai biến thể so sánh được
            assert a["test_start"] == b["test_start"]
            assert a["test_end"] == b["test_end"]

    def test_gap_variants_generated(self):
        from src.checks import build_fold_variants
        cfg = make_cfg()
        cfg.split["gap_variants"] = [0, 7]
        df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=400)})
        out = build_fold_variants(df, cfg)
        assert set(out) == {0, 7}
        assert all(len(v) == cfg.split["n_folds"] for v in out.values())


class TestBatchHomogeneity:
    def test_uniform_batches_pass(self):
        from src.checks import check_batch_homogeneity
        stats = [{"batch": i, "n_rows": 1000, "n_series": 10,
                  "zero_pct": 30.0 + i * 0.5} for i in range(5)]
        out = check_batch_homogeneity(stats)
        assert out.zero_pct.max() - out.zero_pct.min() <= 10

    def test_skewed_batches_flagged(self, capsys):
        """Lô lệch nhau nhiều phải sinh cảnh báo — đây chính là lỗi đã gặp."""
        from src.checks import check_batch_homogeneity
        stats = [{"batch": 0, "n_rows": 6165000, "n_series": 5000, "zero_pct": 22.2},
                 {"batch": 4, "n_rows": 3134929, "n_series": 5000, "zero_pct": 42.0}]
        check_batch_homogeneity(stats)
        assert "CẢNH BÁO" in capsys.readouterr().out


class TestShuffling:
    def test_selection_is_shuffled_but_reproducible(self):
        """Xáo trộn phải phá vỡ thứ tự gốc nhưng vẫn tái lập được với cùng seed."""
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "store_nbr": 1,
            "item_nbr": np.arange(n),
            # first_date tăng dần: mô phỏng thứ tự tự nhiên của dữ liệu
            "first_date": pd.date_range("2014-04-01", periods=n, freq="D"),
        })
        a = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        b = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        pd.testing.assert_frame_equal(a, b)           # tái lập được
        assert not a.item_nbr.is_monotonic_increasing  # đã phá vỡ thứ tự

        # sau xáo trộn, các lô phải đồng nhất về first_date
        means = [a.iloc[i:i + 100].first_date.mean() for i in range(0, n, 100)]
        spread = (max(means) - min(means)).days
        assert spread < 120, f"Các lô vẫn lệch {spread} ngày"


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


# --------------------------------------------------------------------------- #
# Chỉ số đánh giá
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_wape_survives_all_zero_actuals(self):
        """WAPE phải trả nan chứ không được ném lỗi chia cho không."""
        from src.metrics import wape
        assert np.isnan(wape(np.zeros(10), np.ones(10)))

    def test_wape_perfect_forecast_is_zero(self):
        from src.metrics import wape
        y = np.array([0.0, 3, 0, 5, 0])
        assert wape(y, y.copy()) == 0.0

    def test_scale_factor_nan_on_constant_series(self):
        """Chuỗi toàn 0 không có hệ số chuẩn hoá hợp lệ -> phải trả nan."""
        from src.metrics import scale_factor
        assert np.isnan(scale_factor(np.zeros(50)))

    def test_rmsse_excludes_invalid_scale(self):
        from src.metrics import rmsse
        assert np.isnan(rmsse(np.array([1.0, 2]), np.array([1.0, 2]), np.nan))

    def test_evaluate_reports_positive_days_separately(self):
        """Phải tách riêng ngày có nhu cầu, nếu không mô hình dự báo toàn 0
        sẽ trông tốt giả tạo."""
        from src.metrics import evaluate_forecast
        df = pd.DataFrame({
            "store_nbr": 1, "item_nbr": 1,
            "y": [0.0, 0, 0, 10, 0, 8],
            "yhat": [0.0, 0, 0, 0, 0, 0],       # luôn dự báo 0
        })
        res = evaluate_forecast(df)
        assert res["mae"] < res["mae_positive"]   # chỉ số tổng che giấu sai số
        assert res["n_positive"] == 2


class TestBaselines:
    def test_croston_recovers_known_rate(self):
        from src.models import croston
        y = np.zeros(40)
        y[3::4] = 8.0                    # 8 đơn vị mỗi 4 ngày -> 2.0
        assert abs(croston(y, 0.1, "classic") - 2.0) < 1e-6

    def test_sba_applies_bias_correction(self):
        from src.models import croston
        y = np.zeros(40)
        y[3::4] = 8.0
        ratio = croston(y, 0.1, "sba") / croston(y, 0.1, "classic")
        assert abs(ratio - (1 - 0.1 / 2)) < 1e-9

    def test_all_zero_series_returns_zero(self):
        from src.models import croston
        assert croston(np.zeros(30), 0.1, "classic") == 0.0

    def test_tsb_decays_for_discontinued_item(self):
        """TSB phải hạ dự báo khi mã hàng ngừng bán; Croston thì không."""
        from src.models import croston
        y = np.zeros(60)
        y[:20:2] = 5.0                   # chỉ bán trong 20 ngày đầu
        assert croston(y, 0.1, "tsb") < croston(y, 0.1, "classic")


# --------------------------------------------------------------------------- #
# Kiểm định thống kê
# --------------------------------------------------------------------------- #
class TestSignificance:
    def test_dm_detects_clear_difference(self):
        """Khi mô hình A tệ hơn hẳn, DM phải bác bỏ giả thuyết không."""
        from src.significance import diebold_mariano
        rng = np.random.default_rng(0)
        err_a = rng.normal(0, 3.0, 500)     # sai số lớn
        err_b = rng.normal(0, 1.0, 500)     # sai số nhỏ
        res = diebold_mariano(err_a, err_b, horizon=7)
        assert res["p_value"] < 0.01
        assert res["better"] == "B"

    def test_dm_no_difference_when_identical(self):
        from src.significance import diebold_mariano
        rng = np.random.default_rng(1)
        e = rng.normal(0, 1.0, 500)
        res = diebold_mariano(e, e.copy(), horizon=7)
        assert res["better"] == "không khác biệt"

    def test_holm_is_more_conservative_than_raw(self):
        """Hiệu chỉnh Holm phải loại bớt kết luận so với ngưỡng thô."""
        from src.significance import holm_correction
        ps = [0.001, 0.02, 0.03, 0.04, 0.045]
        rejected = holm_correction(ps, alpha=0.05)
        assert rejected[0] is True                 # nhỏ nhất vẫn đạt
        assert sum(rejected) < sum(p < 0.05 for p in ps)

    def test_holm_stops_at_first_failure(self):
        """Holm xét theo thứ tự p tăng dần và dừng ở giá trị đầu tiên không đạt.

        Với [0.9, 0.001]: xét 0.001 trước, đạt ngưỡng 0.05/2 nên bác bỏ; sang
        0.9 thì không đạt nên dừng. Kết quả đúng là [False, True] — thứ tự
        trong danh sách đầu vào không ảnh hưởng tới việc giá trị nào được xét
        trước.
        """
        from src.significance import holm_correction
        assert holm_correction([0.9, 0.001]) == [False, True]
        # cả hai đều lớn -> không bác bỏ cái nào
        assert holm_correction([0.6, 0.4]) == [False, False]

    def test_paired_test_finds_consistent_winner(self):
        from src.significance import paired_test_by_series
        rng = np.random.default_rng(2)
        n = 300
        base = rng.gamma(2, 1, n)
        df = pd.concat([
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "A", "mae": base + 0.3}),
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "B", "mae": base}),
        ])
        res = paired_test_by_series(df, "A", "B")
        assert res["p_wilcoxon"] < 0.001
        assert res["verdict"] == "B tốt hơn"


class TestStage2Objectives:
    def test_all_objectives_registered(self):
        from src.models import STAGE2_OBJECTIVES
        assert set(STAGE2_OBJECTIVES) >= {"squared", "gamma", "absolute",
                                          "poisson", "log_squared"}

    def test_log_transform_roundtrip(self):
        """Biến đổi log rồi hoàn nguyên phải trả về giá trị ban đầu."""
        from src.models import _predict_stage2, STAGE2_OBJECTIVES

        class Stub:
            def predict(self, X):
                return np.log1p(np.array([1.0, 5.0, 20.0]))

        got = _predict_stage2(Stub(), None,
                              STAGE2_OBJECTIVES["log_squared"], shift=False)
        assert np.allclose(got, [1.0, 5.0, 20.0])

    def test_shift_adds_one_back(self):
        from src.models import _predict_stage2, STAGE2_OBJECTIVES

        class Stub:
            def predict(self, X):
                return np.array([0.0, 2.0, 9.0])

        got = _predict_stage2(Stub(), None, STAGE2_OBJECTIVES["squared"],
                              shift=True)
        assert np.allclose(got, [1.0, 3.0, 10.0])

    def test_device_falls_back_to_cpu(self):
        """Yêu cầu CPU phải luôn trả về CPU, không phụ thuộc máy."""
        from src.models import resolve_device
        assert resolve_device("cpu") == "cpu"

    def test_device_auto_returns_valid_value(self):
        from src.models import resolve_device
        assert resolve_device("auto") in ("cpu", "cuda")


# --------------------------------------------------------------------------- #
# Chẩn đoán độ chệch — phát hiện mô hình suy biến
# --------------------------------------------------------------------------- #
class TestBiasDiagnostics:
    def test_bias_ratio_detects_underforecast(self):
        from src.metrics import bias_ratio
        y = np.array([10.0, 20.0, 30.0])
        assert bias_ratio(y, y * 0.5) == pytest.approx(0.5)
        assert bias_ratio(y, y) == pytest.approx(1.0)

    def test_bias_ratio_nan_when_no_demand(self):
        from src.metrics import bias_ratio
        assert np.isnan(bias_ratio(np.zeros(5), np.ones(5)))

    def test_near_zero_rate_flags_degenerate_model(self):
        """Mô hình dự báo toàn số không phải bị phát hiện."""
        from src.metrics import near_zero_rate
        assert near_zero_rate(np.zeros(100)) == 1.0
        assert near_zero_rate(np.full(100, 5.0)) == 0.0

    def test_degenerate_model_has_good_mae_but_bad_bias(self):
        """Đây là bệnh lý cốt lõi: sai số nhỏ nhưng dự báo vô dụng.

        Trên chuỗi có quá nửa số ngày bằng không, mô hình dự báo toàn số không
        đạt MAE tốt hơn mô hình dự báo đúng kỳ vọng — nhưng hoàn toàn không
        dùng được. Chỉ chỉ số độ chệch mới phơi bày điều này.
        """
        from src.metrics import evaluate_forecast
        rng = np.random.default_rng(0)
        n = 5000
        y = np.where(rng.random(n) < 0.65, 0.0, rng.poisson(5, n) + 1.0)

        df_zero = pd.DataFrame({"store_nbr": 1, "item_nbr": 1,
                                "y": y, "yhat": np.zeros(n)})
        df_mean = pd.DataFrame({"store_nbr": 1, "item_nbr": 1,
                                "y": y, "yhat": np.full(n, y.mean())})

        r0, rm = evaluate_forecast(df_zero), evaluate_forecast(df_mean)
        assert r0["mae"] < rm["mae"]            # dự báo 0 có MAE tốt hơn
        assert r0["bias_ratio"] == 0.0          # nhưng chệch hoàn toàn
        assert rm["bias_ratio"] == pytest.approx(1.0, abs=0.01)
        assert r0["near_zero_rate"] == 1.0


class TestFactorialDesign:
    def test_single_stage_objectives_available(self):
        from src.models import SINGLE_STAGE_OBJECTIVES
        assert set(SINGLE_STAGE_OBJECTIVES) >= {"tweedie", "absolute",
                                                "squared"}

    def test_invalid_objective_rejected(self):
        from src.models import fit_single_stage
        df = pd.DataFrame({"store_nbr": [1], "item_nbr": [1],
                           "date": [pd.Timestamp("2015-01-01")],
                           "y": [1.0], "f": [0.5]})
        with pytest.raises(ValueError, match="objective không hợp lệ"):
            fit_single_stage(df, df, ["f"], objective="không_tồn_tại")

    def test_invalid_stage2_rejected(self):
        from src.models import fit_two_stage
        df = pd.DataFrame({"store_nbr": [1], "item_nbr": [1],
                           "date": [pd.Timestamp("2015-01-01")],
                           "y": [1.0], "y_occurrence": [1], "f": [0.5]})
        with pytest.raises(ValueError, match="stage2 không hợp lệ"):
            fit_two_stage(df, df, ["f"], stage2="không_tồn_tại")


class TestEffectSize:
    def test_cohen_d_reported(self):
        """Chênh lệch lớn và ổn định phải cho độ lớn hiệu ứng 'lớn'."""
        from src.significance import paired_test_by_series
        rng = np.random.default_rng(5)
        n = 500
        base = rng.gamma(2, 1, n)
        # chênh lệch có dao động, giống thực tế hơn chênh lệch hằng số
        gap = 2.0 + rng.normal(0, 0.5, n)
        df = pd.concat([
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "A", "mae": base + gap}),
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "B", "mae": base}),
        ])
        res = paired_test_by_series(df, "A", "B")
        assert res["cohen_d"] > 0.8
        assert res["độ_lớn"] == "lớn"

    def test_constant_difference_is_infinite_effect(self):
        """Chênh lệch hằng số nghĩa là hiệu ứng nhất quán tuyệt đối.

        Trả về không ở đây sẽ nói ngược hoàn toàn ý nghĩa: độ lệch chuẩn bằng
        không không phải vì không có khác biệt, mà vì khác biệt lặp lại y hệt
        trên mọi chuỗi.
        """
        from src.significance import paired_test_by_series
        n = 300
        df = pd.concat([
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "A", "mae": np.full(n, 3.0)}),
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "B", "mae": np.full(n, 1.0)}),
        ])
        res = paired_test_by_series(df, "A", "B")
        assert np.isinf(res["cohen_d"]) and res["cohen_d"] > 0
        assert res["độ_lớn"] == "lớn"

    def test_tiny_difference_flagged_as_negligible(self):
        """Chênh lệch nhỏ trên mẫu lớn: p nhỏ nhưng độ lớn không đáng kể."""
        from src.significance import paired_test_by_series
        rng = np.random.default_rng(6)
        n = 20000
        base = rng.gamma(2, 1, n)
        # chênh lệch rất nhỏ so với độ phân tán: đúng tình huống gặp phải
        # trong thực nghiệm, nơi WAPE chỉ lệch vài phần nghìn
        gap = 0.03 + rng.normal(0, 0.5, n)
        df = pd.concat([
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "A", "mae": base + gap}),
            pd.DataFrame({"store_nbr": 1, "item_nbr": np.arange(n),
                          "model_key": "B", "mae": base}),
        ])
        res = paired_test_by_series(df, "A", "B")
        assert res["p_wilcoxon"] < 0.001          # đạt ý nghĩa thống kê
        assert res["độ_lớn"] == "không đáng kể"   # nhưng không đáng kể


class TestModuleIntegrity:
    """Chặn lỗi đã gặp hai lần: định nghĩa hàm nằm sau điểm khởi chạy.

    Python thực thi module từ trên xuống. Khối ``if __name__ == "__main__"``
    gọi ``main()`` ngay tại chỗ, nên bất kỳ hàm nào được định nghĩa PHÍA SAU
    khối đó sẽ chưa tồn tại khi chương trình chạy. Lỗi này không lộ ra lúc
    nạp module mà chỉ vỡ giữa chừng, sau khi thực nghiệm đã chạy hàng chục
    phút — nên đáng để kiểm tra tự động.

    Lưu ý phân biệt: hàm gọi hàm khác nằm dưới nó là HỢP LỆ, vì tên chỉ được
    phân giải lúc gọi chứ không phải lúc định nghĩa. Chỉ vị trí tương đối so
    với điểm khởi chạy mới quan trọng.
    """

    @staticmethod
    def _check(path):
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))

        entry_line = None
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare)
                        and isinstance(test.left, ast.Name)
                        and test.left.id == "__name__"):
                    entry_line = node.lineno
        if entry_line is None:
            return []

        return [n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.lineno > entry_line]

    def test_nothing_defined_after_entry_point(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "src"
        for mod in root.glob("*.py"):
            late = self._check(mod)
            assert not late, (
                f"{mod.name}: {late} được định nghĩa sau khối "
                f"if __name__ == '__main__', nên sẽ chưa tồn tại khi chạy")


class TestModelRecommendation:
    def test_biased_model_excluded_from_recommendation(self):
        """Mô hình lệch quá ngưỡng phải bị loại dù đứng đầu bảng WAPE.

        Tái hiện đúng tình huống gặp trong thực nghiệm: biến thể dùng sai số
        tuyệt đối đạt WAPE tốt nhất nhưng dự báo thiếu 14%, nên không được
        khuyến nghị.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.experiment import _recommend_model
        import tempfile

        results = pd.DataFrame([
            # WAPE tốt nhất nhưng lệch nặng -> phải bị loại
            {"model": "Single-Stage[absolute]", "feature_set": "full",
             "wape": 0.4914, "rmse": 16.01, "rmsse": 0.7124,
             "bias_ratio": 0.8586},
            # cân bằng và tốt trên RMSE -> phải được chọn
            {"model": "Two-Stage[gamma]", "feature_set": "full",
             "wape": 0.5022, "rmse": 15.69, "rmsse": 0.6992,
             "bias_ratio": 0.9838},
            {"model": "Single-Stage", "feature_set": "full",
             "wape": 0.5067, "rmse": 15.72, "rmsse": 0.7017,
             "bias_ratio": 0.9820},
        ])
        with tempfile.TemporaryDirectory() as d:
            got = _recommend_model(results, Path(d), gap=0)
        assert got == "Two-Stage[gamma]"

    def test_falls_back_when_nothing_balanced(self):
        import tempfile
        from pathlib import Path
        from src.experiment import _recommend_model

        results = pd.DataFrame([
            {"model": "A", "feature_set": "full", "wape": 0.5,
             "rmse": 10.0, "rmsse": 0.7, "bias_ratio": 0.5},
        ])
        with tempfile.TemporaryDirectory() as d:
            assert _recommend_model(results, Path(d), gap=0) == ""
