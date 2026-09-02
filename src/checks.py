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


def check_batch_homogeneity(batch_stats: list[dict],
                            max_spread_pct: float = 10.0) -> pd.DataFrame:
    """Cảnh báo nếu các lô lệch nhau quá nhiều về đặc tính dữ liệu.

    Mỗi lô được ghi ra một tệp riêng, và bước huấn luyện thường chỉ nạp một
    phần các tệp đó để tiết kiệm bộ nhớ. Nếu các lô không đồng nhất, mẫu thu
    được sẽ thiên lệch mà không có dấu hiệu nào báo lỗi — dạng hỏng âm thầm.

    Nguyên nhân thường gặp là quên xáo trộn thứ tự chuỗi trước khi chia lô,
    khiến lô đầu gom toàn chuỗi lịch sử dài còn lô cuối toàn chuỗi ra mắt muộn.
    """
    df = pd.DataFrame(batch_stats)
    if len(df) < 2:
        return df

    spread = df.zero_pct.max() - df.zero_pct.min()
    size_ratio = df.n_rows.min() / df.n_rows.max()

    print("\n--- Đồng nhất giữa các lô ---")
    print(df.to_string(index=False))
    print(f"  Chênh lệch zero-demand : {spread:.1f} điểm phần trăm")
    print(f"  Tỷ lệ kích thước nhỏ/lớn: {size_ratio:.2f}")

    if spread > max_spread_pct:
        print(f"  [CẢNH BÁO] các lô lệch quá {max_spread_pct} điểm — "
              f"kiểm tra lại việc xáo trộn chuỗi trước khi chia lô")
    else:
        print("  [đạt] các lô đồng nhất")
    return df


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
def rolling_origin_folds(df: pd.DataFrame, cfg: Config,
                         gap_days: int | None = None,
                         verbose: bool = True) -> list[dict]:
    """Sinh các fold theo kiểu rolling-origin.

    Chuỗi thời gian không được chia ngẫu nhiên: làm vậy là huấn luyện trên
    tương lai để dự báo quá khứ. Mỗi fold ở đây mở rộng dần tập huấn luyện và
    dịch cửa sổ kiểm tra về phía sau, mô phỏng đúng cách mô hình được dùng
    trong thực tế.

    Tham số ``gap_days`` chèn khoảng đệm (embargo) giữa ngày cuối tập huấn
    luyện và ngày đầu tập kiểm tra. Cần hiểu đúng nó đo cái gì:

      ``gap = 0``
          Không có khoảng đệm — mô phỏng kịch bản huấn luyện lại sát thời
          điểm dự báo.

      ``gap = horizon``
          Khoảng đệm bằng đúng horizon, mô phỏng mô hình đã "cũ" đi một
          khoảng thời gian trước khi được dùng để dự báo.

    Lưu ý quan trọng về những gì tham số này KHÔNG kiểm soát: đặc trưng
    (lag, rolling) được sinh một lần trên toàn bộ lưới theo ngày của chính
    mỗi dòng (xem ``features.py``, luôn dịch tối thiểu ``horizon`` ngày so
    với ngày đó), không phụ thuộc ranh giới fold. Vì vậy một dòng kiểm tra ở
    ``test_start`` vẫn mang giá trị lag tính từ dữ liệu thực nằm TRONG
    khoảng đệm (giữa ``train_end`` và ``test_start``), không phải dữ liệu bị
    đóng băng tại ``train_end``. Tham số ``gap`` do đó đo độ "cũ" của tham số
    mô hình đã huấn luyện so với thời điểm đánh giá — không đo việc đặc
    trưng có được cập nhật hay không, vì đặc trưng luôn được cập nhật theo
    đúng ngày của từng dòng. Đây là một thiết kế hợp lệ (nhiều hệ thống sản
    xuất thực tế vận hành đúng kiểu "huấn luyện định kỳ, đặc trưng cập nhật
    liên tục"), nhưng khác với việc "huấn luyện một lần rồi dự báo trọn chu
    kỳ mà hoàn toàn không có thông tin gì mới" — nếu muốn kiểm định đúng
    kịch bản đó thì cần đóng băng đặc trưng tại ``train_end`` hoặc dự báo đệ
    quy, việc này chưa được triển khai.
    """
    s = cfg.split
    gap = s["gap_days"] if gap_days is None else gap_days
    end = df.date.max()
    folds = []

    for i in range(s["n_folds"]):
        offset = (s["n_folds"] - 1 - i) * s["test_days"]
        test_end = end - pd.Timedelta(days=offset)
        test_start = test_end - pd.Timedelta(days=s["test_days"] - 1)
        train_end = test_start - pd.Timedelta(days=gap + 1)

        folds.append({
            "fold": i,
            "gap_days": gap,
            "train_start": df.date.min(),
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    if verbose:
        print(f"\n--- Fold rolling-origin (gap = {gap} ngày) ---")
        for f in folds:
            n_train = (f["train_end"] - f["train_start"]).days + 1
            print(f"  fold {f['fold']}: train {n_train:>4} ngày tới "
                  f"{f['train_end'].date()} | test {f['test_start'].date()} "
                  f"-> {f['test_end'].date()}")
    return folds


def build_fold_variants(df: pd.DataFrame, cfg: Config) -> dict[int, list[dict]]:
    """Sinh nhiều bộ fold ứng với các giá trị gap khác nhau.

    Giá trị gap chỉ ảnh hưởng ranh giới fold chứ không ảnh hưởng đặc trưng,
    nên sinh tất cả biến thể trong cùng một lần chạy thay vì chạy lại toàn bộ
    pipeline cho mỗi giá trị.
    """
    variants = cfg.split.get("gap_variants") or [cfg.split["gap_days"]]
    out = {}
    for gap in variants:
        folds = rolling_origin_folds(df, cfg, gap_days=gap)
        assert_folds_ordered(folds, cfg, gap_days=gap)
        out[gap] = folds
    return out


def assert_folds_ordered(folds: list[dict], cfg: Config,
                         gap_days: int | None = None) -> None:
    """Xác minh không fold nào có tập kiểm tra nằm trước tập huấn luyện."""
    want = cfg.split["gap_days"] if gap_days is None else gap_days
    for f in folds:
        if f["train_end"] >= f["test_start"]:
            raise LeakageError(
                f"Fold {f['fold']}: tập huấn luyện kết thúc "
                f"{f['train_end'].date()} chồng lấn tập kiểm tra bắt đầu "
                f"{f['test_start'].date()}"
            )
        gap = (f["test_start"] - f["train_end"]).days - 1
        if gap < want:
            raise LeakageError(
                f"Fold {f['fold']}: khoảng đệm {gap} ngày nhỏ hơn "
                f"cấu hình {want}"
            )
    print(f"  [đạt] {len(folds)} fold đúng thứ tự, đệm >= {want} ngày")
