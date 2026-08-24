"""Dựng lưới đầy đủ ngày × chuỗi.

Đây là bước không thể bỏ qua. Tệp train.csv của Favorita chỉ lưu những ngày
CÓ phát sinh bán hàng; ngày không bán được không tồn tại dưới dạng giá trị 0
mà đơn giản là thiếu dòng. Hai hệ quả:

  1. Nếu huấn luyện thẳng trên dữ liệu thô, mô hình chỉ nhìn thấy ngày có bán
     và mất hoàn toàn tín hiệu về nhu cầu bằng không — tức mất chính thứ mà
     bài toán này quan tâm.

  2. Nghiêm trọng hơn về mặt kỹ thuật: ``shift(k)`` trên dữ liệu khuyết sẽ lấy
     giá trị của một ngày CÁCH ĐÓ RẤT XA thay vì k ngày trước. Đặc trưng lag
     khi đó mang ý nghĩa hoàn toàn khác với tên gọi của nó.

Vì vậy lưới phải được dựng và kiểm chứng TRƯỚC khi sinh bất kỳ đặc trưng nào.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .config import Config
from .data import iter_sales


class GridIntegrityError(RuntimeError):
    """Lưới không liên tục — dừng ngay vì mọi đặc trưng lag sau đó sẽ sai."""


def build_grid(cfg: Config, series: pd.DataFrame,
               sales: pd.DataFrame) -> pd.DataFrame:
    """Dựng lưới đầy đủ cho một lô chuỗi và gắn doanh số vào.

    Cửa sổ của mỗi chuỗi chạy từ ngày bán đầu tiên của chính nó tới hết kỳ dữ
    liệu, không phải từ đầu kỳ chung. Lý do: những ngày trước khi mặt hàng được
    bày bán tại cửa hàng đó là "chưa kinh doanh", không phải "nhu cầu bằng
    không" — gộp chung sẽ thổi phồng mức độ gián đoạn.

    Parameters
    ----------
    series : DataFrame
        Các chuỗi cần dựng lưới, cần có store_nbr, item_nbr, first_date.
    sales : DataFrame
        Doanh số thô đã lọc sẵn cho đúng các chuỗi này.
    """
    end = pd.Timestamp(series.global_end.iloc[0]) if "global_end" in series \
        else sales.date.max()

    frames = []
    for row in series.itertuples(index=False):
        dates = pd.date_range(row.first_date, end, freq="D")
        frames.append(pd.DataFrame({
            "date": dates,
            "store_nbr": np.int16(row.store_nbr),
            "item_nbr": np.int32(row.item_nbr),
        }))
    grid = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Gộp doanh số vào lưới; ngày không có bản ghi -> nhu cầu bằng 0
    agg = (sales.groupby(["store_nbr", "item_nbr", "date"], as_index=False)
                .agg(unit_sales=("unit_sales", "sum"),
                     onpromotion=("onpromotion", "max")))

    grid = grid.merge(agg, on=["store_nbr", "item_nbr", "date"], how="left")

    # Hàng trả lại (giá trị âm) chiếm 0,006% dữ liệu — cắt về 0 và ghi nhận
    grid["y"] = grid.unit_sales.fillna(0.0).clip(lower=0).astype("float32")
    grid.drop(columns=["unit_sales"], inplace=True)

    # onpromotion: ngày không có bản ghi nghĩa là mặt hàng không được khuyến mãi
    grid["onpromotion"] = _normalise_promo(grid.onpromotion)

    grid.sort_values(["store_nbr", "item_nbr", "date"], inplace=True)
    grid.reset_index(drop=True, inplace=True)
    return grid


def _normalise_promo(col: pd.Series) -> pd.Series:
    """Chuẩn hoá cột khuyến mãi về 0/1.

    Cột gốc là kiểu object với các giá trị True/False/NaN, và tuỳ cách pandas
    đọc file mà có thể là bool Python hoặc chuỗi ký tự.
    """
    mapped = col.map({True: 1, False: 0, "True": 1, "False": 0})
    return mapped.fillna(0).astype("int8")


def assert_grid_complete(grid: pd.DataFrame) -> None:
    """Kiểm chứng lưới liên tục từng ngày cho mọi chuỗi.

    Chạy kiểm tra này trước khi sinh đặc trưng. Nếu lưới thủng, ``shift`` sẽ
    lặng lẽ tạo ra lag sai mà không báo lỗi — dạng hỏng khó phát hiện nhất.
    """
    g = grid.groupby(["store_nbr", "item_nbr"])["date"]
    span = (g.max() - g.min()).dt.days + 1
    count = g.count()

    broken = span[span != count]
    if len(broken):
        raise GridIntegrityError(
            f"{len(broken)} chuỗi có lưới không liên tục. "
            f"Ví dụ: {broken.head(3).to_dict()}. "
            f"Đặc trưng lag sẽ sai nếu tiếp tục."
        )

    dup = grid.duplicated(["store_nbr", "item_nbr", "date"]).sum()
    if dup:
        raise GridIntegrityError(f"Có {dup} dòng trùng khoá (chuỗi, ngày).")


def load_filtered_sales(cfg: Config, series: pd.DataFrame) -> pd.DataFrame:
    """Đọc train.csv MỘT LẦN, chỉ giữ lại các chuỗi đã chọn.

    Đọc lại toàn bộ 125 triệu dòng cho mỗi lô sẽ tốn thời gian gấp nhiều lần
    mà không thu được gì. Sau khi lọc còn khoảng 19 triệu dòng với tập mẫu 25
    nghìn chuỗi, tương đương vài trăm megabyte — hoàn toàn nằm gọn trong bộ
    nhớ, nên giữ luôn ở đó rồi chia lô từ đây.
    """
    keys = set(zip(series.store_nbr.astype("int64"),
                   series.item_nbr.astype("int64")))
    print(f"\nĐọc doanh số cho {len(keys):,} chuỗi đã chọn...")

    parts = []
    total_read = 0
    for chunk in iter_sales(cfg):
        total_read += len(chunk)
        idx = pd.MultiIndex.from_arrays(
            [chunk.store_nbr.astype("int64"), chunk.item_nbr.astype("int64")])
        mask = idx.isin(keys)
        if mask.any():
            parts.append(chunk[mask])
        del chunk, idx
        gc.collect()

    sales = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    mb = sales.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  đã quét {total_read:,} dòng -> giữ {len(sales):,} dòng ({mb:.0f} MB)")
    return sales


def iter_grid_batches(cfg: Config, series: pd.DataFrame,
                      sales: pd.DataFrame):
    """Sinh lưới theo lô chuỗi để giữ bộ nhớ trong tầm kiểm soát.

    Toàn bộ 133 nghìn chuỗi nhân 1.233 ngày cho khoảng 164 triệu dòng, vượt xa
    bộ nhớ khả dụng. Xử lý theo lô cho phép chạy trên máy cấu hình thường mà
    kết quả không đổi.
    """
    batch_size = cfg.batch_size
    n_batches = int(np.ceil(len(series) / batch_size))
    print(f"\nDựng lưới: {len(series):,} chuỗi, {n_batches} lô "
          f"(mỗi lô {batch_size:,} chuỗi)")

    sales_idx = sales.set_index(["store_nbr", "item_nbr"]).sort_index()

    for i in range(n_batches):
        batch = series.iloc[i * batch_size:(i + 1) * batch_size]
        batch_keys = list(zip(batch.store_nbr, batch.item_nbr))

        present = [k for k in batch_keys if k in sales_idx.index]
        sub = (sales_idx.loc[present].reset_index()
               if present else sales.iloc[:0].copy())

        grid = build_grid(cfg, batch, sub)
        assert_grid_complete(grid)

        zero_pct = (grid.y == 0).mean() * 100
        print(f"  lô {i + 1}/{n_batches}: {len(grid):,} dòng | "
              f"zero {zero_pct:.1f}%")

        yield i, grid
        del grid, sub
        gc.collect()
