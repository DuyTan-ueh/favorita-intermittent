"""Sinh đặc trưng, với cơ chế khai báo chống rò rỉ thông tin tương lai.

Ý tưởng trung tâm
-----------------
Mỗi đặc trưng phải tự khai báo nó thuộc loại nào:

  ``KNOWN_IN_ADVANCE``
      Giá trị đã biết trước thời điểm dự báo. Lịch, ngày lễ, và trạng thái
      khuyến mãi đều thuộc nhóm này: doanh nghiệp lên kế hoạch khuyến mãi từ
      trước, nên tại thời điểm lập dự báo ta đã biết ngày mai mặt hàng có được
      giảm giá hay không. Nhóm này được phép dùng giá trị của chính ngày đích.

  ``LAGGED``
      Dẫn xuất từ lịch sử nhu cầu. Chỉ được dùng dữ liệu cách ngày đích ít nhất
      ``horizon`` ngày, vì khi dự báo trước ``horizon`` ngày thì những ngày gần
      hơn thế vẫn chưa xảy ra.

  ``STATIC``
      Thuộc tính không đổi theo thời gian (nhóm hàng, loại cửa hàng). An toàn.

Phân loại này không chỉ để ghi chú. Hàm ``validate_specs`` đối chiếu khai báo
với cấu hình và dừng chương trình nếu phát hiện mâu thuẫn, nên một lỗi rò rỉ
không thể lọt qua chỉ vì người viết quên.

Vì sao khuyến mãi được xếp vào nhóm biết trước
----------------------------------------------
Cột ``onpromotion`` được cuộc thi Favorita cung cấp cho cả giai đoạn huấn
luyện lẫn giai đoạn cần dự báo, ở cùng mức chi tiết ngày/cửa hàng/mã hàng —
đây là căn cứ để xếp nó vào nhóm biết trước. Cần nói rõ giới hạn của khẳng
định này: dữ liệu xác nhận cột này TỒN TẠI cho các ngày tương lai trong cấu
trúc bài toán của cuộc thi, nhưng không chứng minh được quy trình nghiệp vụ
thật của Favorita — ví dụ khuyến mãi được lên kế hoạch trước bao nhiêu ngày,
hay có bị điều chỉnh sát ngày hay không. Nói cách khác, đây là một giả định
được biện minh bởi cách cuộc thi cấu trúc dữ liệu, không phải một sự thật đã
được kiểm chứng về vận hành nội bộ của doanh nghiệp. Ngược lại, việc suy
ngược trạng thái khuyến mãi TỪ doanh số tăng đột biến của chính ngày đó mới
chắc chắn là rò rỉ, vì nó dùng kết quả để suy ra nguyên nhân. Các cột như số
tiền giảm giá thực tế hay doanh thu cũng thuộc loại "chỉ biết sau khi đã
bán", nên không được đưa vào làm đặc trưng.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .config import Config, ConfigError

KEYS = ["store_nbr", "item_nbr"]


class Availability(Enum):
    KNOWN_IN_ADVANCE = "known_in_advance"
    LAGGED = "lagged"
    STATIC = "static"


@dataclass(frozen=True)
class FeatureSpec:
    """Khai báo một đặc trưng: tên, loại, và độ trễ tối thiểu."""
    name: str
    availability: Availability
    min_lag: int = 0
    group: str = "other"

    def __str__(self) -> str:
        return f"{self.name} [{self.availability.value}, lag>={self.min_lag}]"


# --------------------------------------------------------------------------- #
# Khai báo
# --------------------------------------------------------------------------- #
def build_specs(cfg: Config) -> list[FeatureSpec]:
    """Liệt kê toàn bộ đặc trưng sẽ sinh, kèm khai báo tính khả dụng."""
    h = cfg.horizon
    f = cfg.features
    specs: list[FeatureSpec] = []

    for k in cfg.lags:
        specs.append(FeatureSpec(f"lag_{k}", Availability.LAGGED, k, "lag"))

    for w in cfg.windows:
        for stat in f["rolling_stats"]:
            specs.append(FeatureSpec(f"roll_{stat}_{w}",
                                     Availability.LAGGED, h, "rolling"))
        specs.append(FeatureSpec(f"roll_nonzero_rate_{w}",
                                 Availability.LAGGED, h, "rolling"))

    specs.append(FeatureSpec("days_since_last_demand",
                             Availability.LAGGED, h, "intermittency"))
    specs.append(FeatureSpec("demand_count_28",
                             Availability.LAGGED, h, "intermittency"))

    if f["use_calendar"]:
        for name in ["dow", "is_weekend", "day_of_month", "month",
                     "is_month_start", "is_month_end", "payday"]:
            specs.append(FeatureSpec(name, Availability.KNOWN_IN_ADVANCE,
                                     0, "calendar"))

    if f["use_promotion"]:
        specs.append(FeatureSpec("onpromotion",
                                 Availability.KNOWN_IN_ADVANCE, 0, "promotion"))
        specs.append(FeatureSpec("promo_count_next_7",
                                 Availability.KNOWN_IN_ADVANCE, 0, "promotion"))
        # đếm lùi vẫn cần trễ vì dựa trên quá khứ của chính chuỗi
        specs.append(FeatureSpec("promo_count_prev_28",
                                 Availability.LAGGED, h, "promotion"))

    if f["use_holidays"]:
        for name in ["is_holiday", "holiday_type", "days_to_holiday"]:
            specs.append(FeatureSpec(name, Availability.KNOWN_IN_ADVANCE,
                                     0, "holiday"))

    if f["use_static"]:
        for name in ["family", "class", "perishable",
                     "store_type", "store_cluster", "city", "state"]:
            specs.append(FeatureSpec(name, Availability.STATIC, 0, "static"))

    return specs


def validate_specs(cfg: Config, specs: list[FeatureSpec]) -> None:
    """Đối chiếu khai báo với cấu hình; sai thì dừng ngay."""
    h = cfg.horizon
    offenders = [s for s in specs
                 if s.availability is Availability.LAGGED and s.min_lag < h]
    if offenders:
        raise ConfigError(
            "LEAKAGE: các đặc trưng sau khai báo độ trễ nhỏ hơn horizon "
            f"({h}): {[str(s) for s in offenders]}"
        )
    names = [s.name for s in specs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ConfigError(f"Tên đặc trưng bị trùng: {dup}")


def summarise_specs(specs: list[FeatureSpec]) -> pd.DataFrame:
    """Bảng tóm tắt đặc trưng — dùng luôn cho phần Methodology của bài báo."""
    return (pd.DataFrame([{"feature": s.name, "group": s.group,
                           "availability": s.availability.value,
                           "min_lag": s.min_lag} for s in specs])
            .sort_values(["group", "feature"]).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Sinh đặc trưng
# --------------------------------------------------------------------------- #
def add_lag_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Đặc trưng trễ. Lưới phải liên tục thì ``shift`` mới đúng nghĩa."""
    g = df.groupby(KEYS, sort=False)["y"]
    for k in cfg.lags:
        df[f"lag_{k}"] = g.shift(k).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Thống kê trượt, luôn dịch trước ``horizon`` ngày.

    Bước ``shift(horizon)`` là bắt buộc và là lỗi hay gặp nhất khi làm đặc
    trưng chuỗi thời gian: nếu tính trực tiếp, cửa sổ trượt bao gồm cả giá trị
    của chính ngày đích, tức là dùng đáp án để tạo đặc trưng.
    """
    h = cfg.horizon
    g = df.groupby(KEYS, sort=False)["y"]
    base = g.shift(h)                      # <-- điểm mấu chốt
    grouped = base.groupby([df.store_nbr, df.item_nbr], sort=False)

    for w in cfg.windows:
        roll = grouped.rolling(w, min_periods=1)
        for stat in cfg.features["rolling_stats"]:
            vals = getattr(roll, stat)().reset_index(level=[0, 1], drop=True)
            df[f"roll_{stat}_{w}"] = vals.astype("float32")

        nz = (base > 0).astype("float32")
        rate = (nz.groupby([df.store_nbr, df.item_nbr], sort=False)
                  .rolling(w, min_periods=1).mean()
                  .reset_index(level=[0, 1], drop=True))
        df[f"roll_nonzero_rate_{w}"] = rate.astype("float32")

    return df


def add_intermittency_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Đặc trưng riêng cho nhu cầu gián đoạn.

    ``days_since_last_demand`` là biến then chốt của họ phương pháp Croston:
    khoảng cách kể từ lần phát sinh nhu cầu gần nhất. Vẫn phải dịch trước
    ``horizon`` ngày như mọi đặc trưng dẫn xuất từ lịch sử.
    """
    h = cfg.horizon
    shifted = df.groupby(KEYS, sort=False)["y"].shift(h)
    has_demand = (shifted > 0)

    # đánh số nhóm tăng dần mỗi lần có nhu cầu -> đếm ngày trong nhóm hiện tại
    grp = has_demand.groupby([df.store_nbr, df.item_nbr], sort=False).cumsum()
    counter = (has_demand.groupby(
        [df.store_nbr, df.item_nbr, grp], sort=False).cumcount())
    df["days_since_last_demand"] = np.where(
        grp > 0, counter, np.nan).astype("float32")

    df["demand_count_28"] = (
        has_demand.astype("float32")
        .groupby([df.store_nbr, df.item_nbr], sort=False)
        .rolling(28, min_periods=1).sum()
        .reset_index(level=[0, 1], drop=True).astype("float32"))

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng lịch — biết trước hoàn toàn, dùng thẳng ngày đích."""
    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["day_of_month"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["is_month_start"] = d.is_month_start.astype("int8")
    df["is_month_end"] = d.is_month_end.astype("int8")
    # Ecuador trả lương ngày 15 và ngày cuối tháng
    df["payday"] = ((d.day == 15) | d.is_month_end).astype("int8")
    return df


def add_promotion_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Đặc trưng khuyến mãi.

    Lịch khuyến mãi được công bố trước nên số ngày khuyến mãi trong 7 ngày TỚI
    là thông tin hợp lệ tại thời điểm dự báo. Ngược lại, số ngày khuyến mãi
    trong 28 ngày QUA là dẫn xuất từ quá khứ của chuỗi nên vẫn phải dịch trước
    ``horizon`` ngày.
    """
    h = cfg.horizon
    g = df.groupby(KEYS, sort=False)["onpromotion"]

    fwd = (g.apply(lambda s: s[::-1].rolling(7, min_periods=1).sum()[::-1])
           .reset_index(level=[0, 1], drop=True))
    df["promo_count_next_7"] = fwd.astype("float32")

    prev = (df.groupby(KEYS, sort=False)["onpromotion"].shift(h)
              .groupby([df.store_nbr, df.item_nbr], sort=False)
              .rolling(28, min_periods=1).sum()
              .reset_index(level=[0, 1], drop=True))
    df["promo_count_prev_28"] = prev.astype("float32")
    return df


def add_holiday_features(df: pd.DataFrame,
                         holidays: pd.DataFrame) -> pd.DataFrame:
    """Ngày lễ và sự kiện — lịch công bố trước nên an toàn."""
    hol = holidays[holidays.transferred == False].copy()  # noqa: E712
    hol = hol[["date", "type"]].drop_duplicates("date")
    hol["is_holiday"] = 1

    df = df.merge(hol, on="date", how="left")
    df["is_holiday"] = df.is_holiday.fillna(0).astype("int8")
    df["holiday_type"] = df["type"].fillna("None").astype("category")
    df.drop(columns=["type"], inplace=True)

    hol_dates = np.sort(hol.date.values)
    target = df.date.values
    idx = np.searchsorted(hol_dates, target, side="left").clip(
        0, len(hol_dates) - 1)
    df["days_to_holiday"] = (
        (hol_dates[idx] - target) / np.timedelta64(1, "D")).astype("float32")
    return df


def add_static_features(df: pd.DataFrame, items: pd.DataFrame,
                        stores: pd.DataFrame) -> pd.DataFrame:
    """Thuộc tính không đổi theo thời gian."""
    it = items.rename(columns={"family": "family", "class": "class"})
    df = df.merge(it[["item_nbr", "family", "class", "perishable"]],
                  on="item_nbr", how="left")
    st = stores.rename(columns={"type": "store_type",
                                "cluster": "store_cluster"})
    df = df.merge(st[["store_nbr", "city", "state",
                      "store_type", "store_cluster"]],
                  on="store_nbr", how="left")
    for c in ["family", "city", "state", "store_type"]:
        df[c] = df[c].astype("category")
    return df


def engineer(df: pd.DataFrame, cfg: Config, items: pd.DataFrame,
             stores: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Chạy toàn bộ chuỗi sinh đặc trưng theo đúng thứ tự phụ thuộc."""
    f = cfg.features
    df = add_lag_features(df, cfg)
    df = add_rolling_features(df, cfg)
    df = add_intermittency_features(df, cfg)
    if f["use_calendar"]:
        df = add_calendar_features(df)
    if f["use_promotion"]:
        df = add_promotion_features(df, cfg)
    if f["use_holidays"]:
        df = add_holiday_features(df, holidays)
    if f["use_static"]:
        df = add_static_features(df, items, stores)

    # Nhãn cho hai giai đoạn của khung Two-Stage
    df["y_occurrence"] = (df.y > 0).astype("int8")
    df["y_magnitude"] = df.y.where(df.y > 0).astype("float32")
    return df
