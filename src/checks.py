"""Kiểm định tự động chống rò rỉ thông tin, và chia tập theo thời gian.

Khai báo ``FeatureSpec`` cho biết *ý định*; các kiểm định ở đây xác minh *thực
tế*. Hai lớp bổ trợ nhau: khai báo bắt lỗi cấu hình, kiểm định bắt lỗi cài đặt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


class LeakageError(AssertionError):
    """Phát hiện đặc trưng dùng dữ liệu chưa tồn tại tại thời điểm dự báo."""


# --------------------------------------------------------------------------- #
# Kiểm định rò rỉ
# --------------------------------------------------------------------------- #
def truncation_invariance_test(engineer_fn, grid: pd.DataFrame,
                               cfg: Config, cut_date: str | pd.Timestamp,
                               tol: float = 1e-6) -> pd.DataFrame:
    """Kiểm định bất biến khi cắt cụt dữ liệu — phép thử mạnh nhất.

    Nguyên lý: nếu đặc trưng chỉ dùng thông tin quá khứ, thì việc xoá bỏ dữ
    liệu SAU ngày ``cut_date`` không được làm thay đổi giá trị đặc trưng của
    các dòng TRƯỚC ngày đó. Ngược lại, chênh lệch bất kỳ chứng tỏ đặc trưng đã
    nhìn về tương lai.

    Phép thử này bắt được cả những rò rỉ tinh vi mà đọc code khó thấy, chẳng
    hạn quên ``shift`` trong cửa sổ trượt hay dùng thống kê toàn cục.

    Lưu ý: hàm này trả về MỌI đặc trưng có thay đổi, kể cả những đặc trưng
    được phép nhìn tới tương lai như lịch khuyến mãi đã công bố. Việc phân
    định đâu là vi phạm thuộc về ``assert_no_leakage``, nơi kết quả được đối
    chiếu với khai báo ``FeatureSpec``.

    Returns
    -------
    DataFrame liệt kê các đặc trưng lệch nhau. Rỗng nghĩa là đạt.
    """
    cut = pd.Timestamp(cut_date)

    full = engineer_fn(grid.copy(), cfg)
    truncated = engineer_fn(grid[grid.date <= cut].copy(), cfg)

    idx = ["store_nbr", "item_nbr", "date"]
    a = full[full.date <= cut].set_index(idx).sort_index()
    b = truncated.set_index(idx).sort_index()

    common_rows = a.index.intersection(b.index)
    a, b = a.loc[common_rows], b.loc[common_rows]

    rows = []
    for col in a.columns.intersection(b.columns):
        if not pd.api.types.is_numeric_dtype(a[col]):
            continue
        x, y = a[col].to_numpy("float64"), b[col].to_numpy("float64")
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(x - y, nan=np.inf)))
        n_bad = int((diff > tol).sum())
        if n_bad:
            rows.append({"feature": col, "n_mismatch": n_bad,
                         "max_abs_diff": float(diff[np.isfinite(diff)].max()
                                               if np.isfinite(diff).any()
                                               else np.inf)})

    return pd.DataFrame(rows)


def assert_no_leakage(engineer_fn, grid: pd.DataFrame, cfg: Config,
                      cut_date: str | pd.Timestamp, specs=None) -> None:
    """Chạy kiểm định bất biến, đối chiếu kết quả với khai báo đặc trưng.

    Không phải mọi thay đổi đều là vi phạm. Đặc trưng thuộc nhóm biết trước —
    điển hình là số ngày khuyến mãi trong tuần tới — hợp lệ khi nhìn về tương
    lai, vì lịch khuyến mãi được công bố từ trước và đã nằm trong tay người
    lập dự báo. Cắt cụt dữ liệu tất nhiên làm những đặc trưng này đổi giá trị
    ở vùng sát mốc cắt, nhưng đó là tạo tác của phép thử chứ không phải lỗi.

    Vi phạm thực sự là khi đặc trưng thuộc nhóm dẫn xuất từ lịch sử nhu cầu
    lại thay đổi — chứng tỏ nó đang đọc dữ liệu chưa tồn tại.
    """
    from .features import Availability, build_specs

    if specs is None:
        specs = build_specs(cfg)
    lagged = {s.name for s in specs if s.availability is Availability.LAGGED}

    changed = truncation_invariance_test(engineer_fn, grid, cfg, cut_date)
    if not len(changed):
        print(f"  [đạt] không đặc trưng nào đổi giá trị khi cắt tại "
              f"{pd.Timestamp(cut_date).date()}")
        return

    violations = changed[changed.feature.isin(lagged)]
    expected = changed[~changed.feature.isin(lagged)]

    if len(expected):
        print(f"  [bỏ qua] {len(expected)} đặc trưng nhóm biết trước thay đổi "
              f"gần mốc cắt (hợp lệ): {', '.join(expected.feature)}")

    if len(violations):
        raise LeakageError(
            "Đặc trưng dẫn xuất từ lịch sử nhu cầu thay đổi giá trị khi dữ "
            "liệu tương lai bị xoá, tức đang đọc thông tin chưa tồn tại:\n"
            f"{violations.to_string(index=False)}"
        )
    print(f"  [đạt] không đặc trưng trễ nào rò rỉ tại "
          f"{pd.Timestamp(cut_date).date()}")


def check_feature_nulls(df: pd.DataFrame, cfg: Config,
                        max_null_rate: float = 0.5) -> pd.DataFrame:
    """Cảnh báo đặc trưng thiếu giá trị quá nhiều.

    Đặc trưng trễ luôn thiếu ở đầu mỗi chuỗi, đó là bình thường. Nhưng tỷ lệ
    thiếu vượt ngưỡng cho thấy cửa sổ quá dài so với độ dài chuỗi.
    """
    feat_cols = [c for c in df.columns
                 if c not in ("y", "y_occurrence", "y_magnitude",
                              "date", "store_nbr", "item_nbr")]
    rates = df[feat_cols].isna().mean().sort_values(ascending=False)
    flagged = rates[rates > max_null_rate]
    if len(flagged):
        print(f"  [cảnh báo] {len(flagged)} đặc trưng thiếu > "
              f"{max_null_rate:.0%} giá trị:")
        print(flagged.head(10).to_string())
    return rates.to_frame("null_rate")


# --------------------------------------------------------------------------- #
# Chia tập theo thời gian
# --------------------------------------------------------------------------- #
def rolling_origin_folds(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """Sinh các fold theo kiểu rolling-origin.

    Chuỗi thời gian không được chia ngẫu nhiên: làm vậy là huấn luyện trên
    tương lai để dự báo quá khứ. Mỗi fold ở đây mở rộng dần tập huấn luyện và
    dịch cửa sổ kiểm tra về phía sau, mô phỏng đúng cách mô hình được dùng
    trong thực tế.

    Tham số ``gap_days`` chèn khoảng đệm giữa huấn luyện và kiểm tra. Nên đặt
    bằng ``horizon`` nếu muốn chặt chẽ tuyệt đối, tránh việc dòng cuối tập
    huấn luyện và dòng đầu tập kiểm tra chia sẻ cùng cửa sổ lịch sử.
    """
    s = cfg.split
    end = df.date.max()
    folds = []

    for i in range(s["n_folds"]):
        offset = (s["n_folds"] - 1 - i) * s["test_days"]
        test_end = end - pd.Timedelta(days=offset)
        test_start = test_end - pd.Timedelta(days=s["test_days"] - 1)
        train_end = test_start - pd.Timedelta(days=s["gap_days"] + 1)

        folds.append({
            "fold": i,
            "train_start": df.date.min(),
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    print("\n--- Các fold rolling-origin ---")
    for f in folds:
        print(f"  fold {f['fold']}: train tới {f['train_end'].date()} | "
              f"test {f['test_start'].date()} -> {f['test_end'].date()}")
    return folds


def assert_folds_ordered(folds: list[dict], cfg: Config) -> None:
    """Xác minh không fold nào có tập kiểm tra nằm trước tập huấn luyện."""
    for f in folds:
        if f["train_end"] >= f["test_start"]:
            raise LeakageError(
                f"Fold {f['fold']}: tập huấn luyện kết thúc "
                f"{f['train_end'].date()} chồng lấn tập kiểm tra bắt đầu "
                f"{f['test_start'].date()}"
            )
        gap = (f["test_start"] - f["train_end"]).days - 1
        if gap < cfg.split["gap_days"]:
            raise LeakageError(
                f"Fold {f['fold']}: khoảng đệm {gap} ngày nhỏ hơn "
                f"cấu hình {cfg.split['gap_days']}"
            )
    print("  [đạt] mọi fold đúng thứ tự thời gian")
