"""Đọc dữ liệu thô và chọn tập chuỗi đưa vào thực nghiệm.

Chia làm hai việc tách bạch:
  1. ``load_*``  — đọc file, ép kiểu tiết kiệm bộ nhớ
  2. ``select_series`` — lọc theo tiêu chí, lấy mẫu phân tầng có thể tái lập
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Ngưỡng phân loại Syntetos-Boylan, dùng thống nhất toàn dự án
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

SALES_DTYPES = {
    "store_nbr": "int8",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "object",
}
SALES_COLS = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]


# --------------------------------------------------------------------------- #
# Đọc file
# --------------------------------------------------------------------------- #
def load_items(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_dir / "items.csv")


def load_stores(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_dir / "stores.csv")


def load_holidays(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_dir / "holidays_events.csv",
                       parse_dates=["date"])


def iter_sales(cfg: Config, chunksize: int = 10_000_000):
    """Đọc train.csv theo lô — 125 triệu dòng không vừa bộ nhớ Kaggle."""
    start = pd.Timestamp(cfg.data["start_date"])
    end = cfg.data.get("end_date")
    end = pd.Timestamp(end) if end else None

    reader = pd.read_csv(cfg.raw_dir / "train.csv", usecols=SALES_COLS,
                         dtype=SALES_DTYPES, parse_dates=["date"],
                         chunksize=chunksize)
    for chunk in reader:
        chunk = chunk[chunk.date >= start]
        if end is not None:
            chunk = chunk[chunk.date <= end]
        if not chunk.empty:
            yield chunk


# --------------------------------------------------------------------------- #
# Thống kê theo chuỗi
# --------------------------------------------------------------------------- #
def compute_series_stats(cfg: Config) -> pd.DataFrame:
    """Tính thống kê mô tả cho từng chuỗi (store, item).

    Lưu ý về cách tính tỷ lệ zero-demand: train.csv KHÔNG chứa dòng cho những
    ngày không bán được, nên không thể đếm trực tiếp. Số ngày có bán được suy
    ra từ số bản ghi, còn tổng số ngày lấy từ độ dài cửa sổ quan sát. Cách này
    cho kết quả tương đương việc dựng lưới đầy đủ nhưng không tốn bộ nhớ.
    """
    acc: dict[tuple[int, int], list] = {}
    gmin = gmax = None

    for chunk in iter_sales(cfg):
        cmin, cmax = chunk.date.min(), chunk.date.max()
        gmin = cmin if gmin is None else min(gmin, cmin)
        gmax = cmax if gmax is None else max(gmax, cmax)

        chunk = chunk.assign(
            _pos=(chunk.unit_sales > 0).astype("int32"),
            _v=chunk.unit_sales.clip(lower=0),
        )
        chunk["_v2"] = chunk["_v"] ** 2

        grouped = chunk.groupby(["store_nbr", "item_nbr"], sort=False).agg(
            n_pos=("_pos", "sum"), sum_pos=("_v", "sum"),
            sumsq_pos=("_v2", "sum"),
            dmin=("date", "min"), dmax=("date", "max"))

        for key, row in zip(grouped.index.values,
                            grouped.itertuples(index=False)):
            slot = acc.get(key)
            if slot is None:
                acc[key] = [row.n_pos, row.sum_pos, row.sumsq_pos,
                            row.dmin, row.dmax]
            else:
                slot[0] += row.n_pos
                slot[1] += row.sum_pos
                slot[2] += row.sumsq_pos
                slot[3] = min(slot[3], row.dmin)
                slot[4] = max(slot[4], row.dmax)

        del chunk, grouped
        gc.collect()

    keys = np.array(list(acc.keys()))
    vals = np.array([v[:3] for v in acc.values()], dtype="float64")

    stats = pd.DataFrame({
        "store_nbr": keys[:, 0].astype("int16"),
        "item_nbr": keys[:, 1].astype("int32"),
        "n_pos": vals[:, 0],
        "sum_pos": vals[:, 1],
        "sumsq_pos": vals[:, 2],
        "first_date": pd.to_datetime([v[3] for v in acc.values()]),
        "last_date": pd.to_datetime([v[4] for v in acc.values()]),
    })

    stats["global_start"] = gmin
    stats["global_end"] = gmax
    stats["days_active"] = (stats.last_date - stats.first_date).dt.days + 1
    stats["zero_pct"] = (1 - stats.n_pos / stats.days_active).clip(lower=0) * 100
    stats["adi"] = stats.days_active / stats.n_pos.replace(0, np.nan)

    mean_pos = stats.sum_pos / stats.n_pos.replace(0, np.nan)
    var_pos = stats.sumsq_pos / stats.n_pos.replace(0, np.nan) - mean_pos ** 2
    stats["mean_demand"] = mean_pos
    stats["cv2"] = var_pos.clip(lower=0) / mean_pos ** 2
    stats["pattern"] = classify_pattern(stats.adi, stats.cv2)

    return stats


def classify_pattern(adi: pd.Series, cv2: pd.Series) -> pd.Series:
    """Phân loại Syntetos-Boylan thành 4 nhóm mẫu nhu cầu."""
    sparse = adi >= ADI_THRESHOLD
    volatile = cv2 >= CV2_THRESHOLD
    out = pd.Series("Unclassified", index=adi.index, dtype="object")
    valid = adi.notna() & cv2.notna()
    out[valid & ~sparse & ~volatile] = "Smooth"
    out[valid & ~sparse & volatile] = "Erratic"
    out[valid & sparse & ~volatile] = "Intermittent"
    out[valid & sparse & volatile] = "Lumpy"
    return out


# --------------------------------------------------------------------------- #
# Chọn tập chuỗi
# --------------------------------------------------------------------------- #
def select_series(cfg: Config, stats: pd.DataFrame) -> pd.DataFrame:
    """Lọc theo tiêu chí rồi lấy mẫu phân tầng.

    Lọc nhằm hai mục đích trái chiều nhau, nên phải cân bằng: giữ đủ quan sát
    dương để giai đoạn hồi quy huấn luyện được, nhưng không siết chặt tới mức
    loại mất chính những chuỗi gián đoạn — vốn là đối tượng nghiên cứu.
    """
    d = cfg.data
    end = stats.global_end.iloc[0]
    cutoff = end - pd.Timedelta(days=d["require_active_until_end"])

    mask = (
        (stats.days_active >= d["min_days_active"])
        & (stats.n_pos >= d["min_positive_days"])
        & (stats.last_date >= cutoff)
    )
    kept = stats[mask].copy()

    log = pd.DataFrame([
        ("Ban đầu", len(stats)),
        (f"days_active >= {d['min_days_active']}",
         int((stats.days_active >= d["min_days_active"]).sum())),
        (f"+ n_pos >= {d['min_positive_days']}",
         int(((stats.days_active >= d["min_days_active"])
              & (stats.n_pos >= d["min_positive_days"])).sum())),
        (f"+ còn bán trong {d['require_active_until_end']} ngày cuối", len(kept)),
    ], columns=["Bước lọc", "Số chuỗi"])
    log["Còn lại %"] = (log["Số chuỗi"] / len(stats) * 100).round(1)

    print("\n--- Waterfall lọc chuỗi ---")
    print(log.to_string(index=False))

    if cfg.sampling["enabled"] and len(kept) > cfg.sampling["n_series"]:
        kept = _stratified_sample(kept, cfg)

    # Xáo trộn trước khi chia lô. Thứ tự tự nhiên của dữ liệu phản ánh thời
    # điểm chuỗi xuất hiện lần đầu, nên nếu giữ nguyên thì lô đầu toàn chuỗi
    # có lịch sử dài (ít ngày không bán) còn lô cuối toàn chuỗi ra mắt muộn
    # (nhiều ngày không bán). Khi đó mỗi tệp kết quả là một mẫu thiên lệch —
    # nguy hiểm vì bước huấn luyện sau này thường chỉ nạp một phần tệp để
    # tiết kiệm bộ nhớ, và sự thiên lệch đó không hề báo lỗi.
    kept = kept.sample(frac=1.0, random_state=cfg.seed)

    print(f"\nTập cuối: {len(kept):,} chuỗi | "
          f"zero-demand TB {kept.zero_pct.mean():.1f}%")
    print(kept.pattern.value_counts().to_string())
    return kept.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Lấy mẫu giữ nguyên tỷ lệ 4 nhóm mẫu nhu cầu.

    Phân tầng quan trọng vì RQ3 so sánh hiệu quả mô hình giữa các nhóm — lấy
    mẫu ngẫu nhiên đơn thuần có thể làm nhóm nhỏ teo lại tới mức không kiểm
    định được.
    """
    n = cfg.sampling["n_series"]
    key = cfg.sampling["stratify_by"]
    frac = n / len(df)

    # Lấy mẫu theo chỉ số thay vì groupby.apply: apply sẽ loại bỏ cột dùng làm
    # khoá nhóm ở các phiên bản pandas mới, khiến cột 'pattern' biến mất khỏi
    # kết quả — trong khi RQ3 lại cần chính cột đó để phân tầng.
    picked = []
    rng_seed = cfg.seed
    for value, group in df.groupby(key, sort=False):
        take = max(1, int(round(len(group) * frac)))
        take = min(take, len(group))
        picked.append(group.sample(n=take, random_state=rng_seed))

    sampled = pd.concat(picked).sort_index()
    print(f"\nLấy mẫu phân tầng theo '{key}': "
          f"{len(df):,} -> {len(sampled):,} chuỗi")
    return sampled


def _eligible_keys(cfg: Config, stats: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """Áp đúng bộ lọc điều kiện (không lấy mẫu) và trả về tập khoá + bảng.

    Tách riêng bước này khỏi ``select_series`` vì ``select_series`` còn gọi
    thêm ``_stratified_sample`` khi bật lấy mẫu. Đo độ ổn định của tiêu chí
    lọc mà lại đi qua bước lấy mẫu ngẫu nhiên là sai: hai lần gọi
    ``DataFrame.sample(n, random_state=...)`` trên hai bảng có cùng nội dung
    nhưng THỨ TỰ DÒNG khác nhau (hoàn toàn có thể xảy ra, vì mỗi lần gọi
    ``compute_series_stats`` tích luỹ khoá vào một dict độc lập) sẽ chọn ra
    những phần tử khác nhau, dù tiêu chí lọc và tập đủ điều kiện giống hệt
    nhau. Đã kiểm chứng trực tiếp: hai tập giống hệt 100% nhưng khác thứ tự
    dòng cho Jaccard sau lấy mẫu chỉ 18% — đủ để một kết luận sai hoàn toàn
    về mức độ ổn định của việc chọn quần thể.
    """
    d = cfg.data
    cutoff = stats.global_end.iloc[0] - pd.Timedelta(
        days=d["require_active_until_end"])
    mask = (
        (stats.days_active >= d["min_days_active"])
        & (stats.n_pos >= d["min_positive_days"])
        & (stats.last_date >= cutoff)
    )
    eligible = stats[mask].copy()
    keys = set(zip(eligible.store_nbr, eligible.item_nbr))
    return keys, eligible


def selection_sensitivity_check(cfg: Config, full_stats: pd.DataFrame,
                                early_cutoff: str | pd.Timestamp) -> dict:
    """Đo mức độ nghiêm trọng thật của việc chọn chuỗi dựa trên toàn timeline.

    Tiêu chí chọn chuỗi (``select_series``) và phân loại mẫu nhu cầu
    (``classify_pattern``) đều dùng thống kê tính trên TOÀN BỘ khoảng thời
    gian nghiên cứu, kể cả phần nằm sau ``train_end`` của fold sớm nhất. Đây
    là một dạng chọn quần thể phụ thuộc tương lai (future-dependent cohort
    selection). Nó không làm sai số liệu huấn luyện của bất kỳ mô hình nào
    (các nhãn này không được dùng làm đặc trưng đầu vào — chỉ dùng để lọc
    quần thể và phân tầng kết quả), nhưng ảnh hưởng tới việc quần thể nghiên
    cứu có đại diện cho "series khả dụng nếu chỉ biết thông tin tại một mốc
    sớm hơn" hay không.

    Hàm này so sánh THEO HAI TẦNG TÁCH BIỆT, không gộp chung:

      1. Chồng lấp ELIGIBILITY (trước khi lấy mẫu) — đây là phép đo chính,
         phản ánh đúng câu hỏi "tiêu chí lọc có phụ thuộc thông tin tương lai
         theo cách ảnh hưởng thực chất không". Không đi qua bước lấy mẫu nên
         không bị nhiễu bởi thứ tự dòng ngẫu nhiên.
      2. Chồng lấp SAMPLE (sau khi lấy mẫu, dùng ``select_series`` đầy đủ) —
         chỉ mang tính tham khảo, cho biết tập 25.000 chuỗi cụ thể thay đổi
         ra sao, nhưng KHÔNG được dùng để kết luận về bản chất tiêu chí lọc.

    Đây là một lần quét dữ liệu bổ sung (tốn thời gian tương đương một lần
    chạy ``compute_series_stats``), không phải một thực nghiệm huấn luyện
    mô hình.
    """
    import copy

    cfg_early = copy.deepcopy(cfg)
    cfg_early.data = dict(cfg.data)
    cfg_early.data["end_date"] = str(pd.Timestamp(early_cutoff).date())

    print(f"Tính lại thống kê chỉ dùng dữ liệu tới {early_cutoff} "
          f"(giả lập thông tin có tại fold sớm nhất)...")
    early_stats = compute_series_stats(cfg_early)

    # ---- TẦNG 1 (chính): chồng lấp ELIGIBILITY, trước sampling ----
    key_full_elig, elig_full = _eligible_keys(cfg, full_stats)
    key_early_elig, elig_early = _eligible_keys(cfg, early_stats)

    overlap_elig = key_full_elig & key_early_elig
    union_elig = key_full_elig | key_early_elig
    eligibility_jaccard = (len(overlap_elig) / len(union_elig)
                           if union_elig else float("nan"))

    # Đồng thuận nhãn pattern, tính trên phần eligibility chung — cũng
    # không đi qua sampling.
    common_elig = elig_full.merge(
        elig_early[["store_nbr", "item_nbr", "pattern"]],
        on=["store_nbr", "item_nbr"], suffixes=("_full", "_early"))
    pattern_agree_elig = ((common_elig.pattern_full
                          == common_elig.pattern_early).mean()
                          if len(common_elig) else float("nan"))

    # ---- TẦNG 2 (tham khảo): chồng lấp SAMPLE, sau sampling ----
    full_sampled = select_series(cfg, full_stats.copy())
    early_sampled = select_series(cfg, early_stats.copy())
    key_full_s = set(zip(full_sampled.store_nbr, full_sampled.item_nbr))
    key_early_s = set(zip(early_sampled.store_nbr, early_sampled.item_nbr))
    sample_jaccard = (len(key_full_s & key_early_s)
                      / len(key_full_s | key_early_s)
                      if (key_full_s | key_early_s) else float("nan"))

    result = {
        "early_cutoff": str(early_cutoff),
        "n_eligible_full_period": len(key_full_elig),
        "n_eligible_early_only": len(key_early_elig),
        "eligibility_jaccard_overlap": eligibility_jaccard,
        "n_eligible_common": len(common_elig),
        "pattern_label_agreement": pattern_agree_elig,
        "sample_jaccard_overlap_INFO_ONLY": sample_jaccard,
    }

    print("\n--- TẦNG 1 (chính): chồng lấp ELIGIBILITY, trước sampling ---")
    print(f"  Số chuỗi đủ điều kiện (toàn kỳ)      : "
          f"{result['n_eligible_full_period']:,}")
    print(f"  Số chuỗi đủ điều kiện (tới "
          f"{early_cutoff}): {result['n_eligible_early_only']:,}")
    print(f"  Chồng lấp eligibility (Jaccard)       : "
          f"{eligibility_jaccard:.1%}")
    print(f"  Đồng thuận nhãn pattern (trên phần chung, "
          f"{len(common_elig):,} chuỗi): {pattern_agree_elig:.1%}")

    print("\n--- TẦNG 2 (chỉ để tham khảo): chồng lấp SAMPLE sau lấy mẫu ---")
    print(f"  Chồng lấp sample (Jaccard)            : {sample_jaccard:.1%}")
    print("  Lưu ý: số này có thể THẤP hơn eligibility Jaccard đáng kể chỉ")
    print("  do thứ tự dòng ngẫu nhiên giữa hai lần tính thống kê, KHÔNG")
    print("  phản ánh riêng độ ổn định của tiêu chí lọc. Không dùng số này")
    print("  để kết luận về bản chất selection.")

    if eligibility_jaccard > 0.9 and pattern_agree_elig > 0.9:
        print("\n  [đánh giá] Chồng lấp eligibility rất cao — dùng thông tin"
              " toàn kỳ hay\n  chỉ tới fold sớm nhất cho kết quả gần như "
              "nhau. Đây chủ yếu là vấn đề lý\n  thuyết, ảnh hưởng thực tế "
              "nhỏ.")
    elif eligibility_jaccard > 0.7 and pattern_agree_elig > 0.7:
        print("\n  [đánh giá] Chồng lấp khá cao nhưng không hoàn toàn — nên"
              " nêu rõ giới\n  hạn này trong Methodology, không cần viết "
              "lại pipeline.")
    else:
        print("\n  [CẢNH BÁO] Chồng lấp eligibility thấp — việc chọn theo "
              "toàn kỳ ảnh\n  hưởng đáng kể tới quần thể nghiên cứu. Nên cân"
              " nhắc viết lại theo\n  hướng per-fold trước khi công bố RQ3.")

    return result
