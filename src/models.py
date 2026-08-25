"""Các mô hình dự báo: baseline cổ điển, một giai đoạn, và Two-Stage.

Ba nhóm mô hình tương ứng ba mốc so sánh của RQ1:

  Baseline cổ điển — Croston, SBA, TSB
      Chuẩn mực lâu đời của bài toán nhu cầu gián đoạn. Croston tách chuỗi
      thành hai thành phần, kích thước đơn và khoảng cách giữa các đơn, rồi
      làm trơn từng thành phần riêng. SBA hiệu chỉnh độ chệch đã biết của
      Croston. TSB thay việc cập nhật theo khoảng cách bằng cập nhật theo xác
      suất, nên phản ứng tốt hơn với mã hàng đang ngừng bán.

  Một giai đoạn — XGBoost hồi quy trực tiếp
      Dự báo thẳng con số, không tách bài toán. Đây là mốc so sánh quan trọng
      nhất: nếu Two-Stage không vượt được nó thì việc tách làm hai bước không
      mang lại giá trị.

  Two-Stage — phân loại rồi hồi quy
      Giai đoạn một ước lượng xác suất có phát sinh đơn. Giai đoạn hai ước
      lượng số lượng, và chỉ được huấn luyện trên những quan sát thực sự có
      đơn. Dự báo cuối là tích của hai đại lượng, tức kỳ vọng có điều kiện.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["store_nbr", "item_nbr"]


# --------------------------------------------------------------------------- #
# Baseline cổ điển
# --------------------------------------------------------------------------- #
def croston(y: np.ndarray, alpha: float = 0.1,
            variant: str = "classic") -> float:
    """Họ phương pháp Croston, trả về dự báo cho mọi bước tương lai.

    Cả ba biến thể đều cho một giá trị không đổi cho toàn bộ chân trời dự báo
    — đó là đặc tính vốn có của phương pháp, không phải giản lược.

    Parameters
    ----------
    variant : {"classic", "sba", "tsb"}
        ``classic`` giữ nguyên công thức gốc; ``sba`` nhân thêm hệ số hiệu
        chỉnh độ chệch; ``tsb`` cập nhật xác suất có nhu cầu ở mọi chu kỳ thay
        vì chỉ khi có đơn.
    """
    y = np.asarray(y, dtype="float64")
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0

    if variant == "tsb":
        return _tsb(y, alpha)

    sizes = y[nz]
    intervals = np.diff(np.concatenate([[-1], nz]))

    z = sizes[0]          # kích thước đơn đã làm trơn
    x = intervals[0]      # khoảng cách giữa các đơn đã làm trơn
    for size, gap in zip(sizes[1:], intervals[1:]):
        z += alpha * (size - z)
        x += alpha * (gap - x)

    if x <= 0:
        return 0.0
    forecast = z / x
    if variant == "sba":
        forecast *= (1 - alpha / 2)      # hiệu chỉnh độ chệch Syntetos-Boylan
    return float(forecast)


def _tsb(y: np.ndarray, alpha: float, beta: float = 0.1) -> float:
    """Biến thể Teunter-Syntetos-Babai.

    Khác biệt cốt lõi: xác suất có nhu cầu được cập nhật ở MỌI chu kỳ, kể cả
    chu kỳ không phát sinh đơn. Nhờ vậy dự báo giảm dần khi mã hàng ngừng bán,
    trong khi Croston gốc giữ nguyên mức cũ vô thời hạn.
    """
    nz = np.flatnonzero(y > 0)
    z = float(y[nz[0]])
    p = 1.0 / (nz[0] + 1)

    for t in range(len(y)):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            p += beta * (1 - p)
        else:
            p += beta * (0 - p)
    return float(z * p)


def fit_predict_baseline(train: pd.DataFrame, test: pd.DataFrame,
                         variant: str, alpha: float = 0.1) -> pd.DataFrame:
    """Chạy baseline cho từng chuỗi, trả về khung dự báo cho tập kiểm tra."""
    preds = {}
    for key, g in train.sort_values("date").groupby(KEYS, sort=False):
        preds[key] = croston(g["y"].to_numpy(), alpha=alpha, variant=variant)

    out = test[KEYS + ["date", "y"]].copy()
    out["yhat"] = [preds.get((s, i), 0.0) for s, i
                   in zip(out.store_nbr, out.item_nbr)]
    return out


# --------------------------------------------------------------------------- #
# Mô hình học máy
# --------------------------------------------------------------------------- #
def _xgb_params(objective: str, seed: int) -> dict:
    base = {
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",       # cần cho dữ liệu quy mô lớn
        "random_state": seed,
        "n_jobs": -1,
    }
    base["objective"] = objective
    return base


def fit_single_stage(train: pd.DataFrame, test: pd.DataFrame,
                     feature_cols: list[str], seed: int = 42,
                     n_estimators: int = 300) -> pd.DataFrame:
    """Hồi quy trực tiếp số lượng, không tách bài toán.

    Dùng hàm mất mát Tweedie thay vì bình phương sai số: phân phối Tweedie có
    khối xác suất tại không cộng với phần liên tục dương, đúng dạng dữ liệu
    nhu cầu gián đoạn. Dùng bình phương sai số ở đây sẽ tạo mốc so sánh yếu
    một cách không cần thiết.
    """
    import xgboost as xgb

    params = _xgb_params("reg:tweedie", seed)
    params["tweedie_variance_power"] = 1.3

    model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
    model.fit(train[feature_cols], train["y"], verbose=False)

    out = test[KEYS + ["date", "y"]].copy()
    out["yhat"] = np.clip(model.predict(test[feature_cols]), 0, None)
    return out, model


def fit_two_stage(train: pd.DataFrame, test: pd.DataFrame,
                  feature_cols: list[str], seed: int = 42,
                  n_estimators: int = 300) -> tuple:
    """Khung hai giai đoạn: phân loại khả năng có đơn, rồi hồi quy số lượng.

    Giai đoạn hai CHỈ được huấn luyện trên các quan sát có nhu cầu dương. Đây
    là điểm mấu chốt của phương pháp: mô hình số lượng không bị hàng loạt giá
    trị bằng không kéo về không, nên học được phân phối số lượng thật sự.

    Dự báo cuối là tích của xác suất và kỳ vọng có điều kiện, tức kỳ vọng
    không điều kiện của nhu cầu.
    """
    import xgboost as xgb

    # --- Giai đoạn 1: có phát sinh đơn hay không ---
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        eval_metric="logloss",
        **_xgb_params("binary:logistic", seed))
    clf.fit(train[feature_cols], train["y_occurrence"], verbose=False)

    # --- Giai đoạn 2: số lượng, chỉ trên ngày có đơn ---
    pos = train[train.y_occurrence == 1]
    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        **_xgb_params("reg:squarederror", seed))
    reg.fit(pos[feature_cols], pos["y"], verbose=False)

    prob = clf.predict_proba(test[feature_cols])[:, 1]
    magnitude = np.clip(reg.predict(test[feature_cols]), 0, None)

    out = test[KEYS + ["date", "y", "y_occurrence"]].copy()
    out["prob_occurrence"] = prob
    out["pred_magnitude"] = magnitude
    out["yhat"] = prob * magnitude          # kỳ vọng không điều kiện
    return out, (clf, reg)


# --------------------------------------------------------------------------- #
# Nhóm đặc trưng cho nghiên cứu loại trừ (RQ2)
# --------------------------------------------------------------------------- #
FEATURE_SETS = {
    "historical": ["lag", "rolling", "intermittency"],
    "hist_calendar": ["lag", "rolling", "intermittency", "calendar"],
    "hist_cal_holiday": ["lag", "rolling", "intermittency", "calendar",
                         "holiday"],
    "full": ["lag", "rolling", "intermittency", "calendar", "holiday",
             "promotion", "static"],
}


def select_features(specs: pd.DataFrame, groups: list[str]) -> list[str]:
    """Chọn cột đặc trưng theo nhóm, phục vụ nghiên cứu loại trừ.

    Bốn cấu hình trong ``FEATURE_SETS`` được thiết kế để cô lập đóng góp của
    từng nhóm: chênh lệch giữa ``hist_cal_holiday`` và ``full`` chính là phần
    đóng góp của đặc trưng khuyến mãi, tức câu trả lời định lượng cho RQ2.
    """
    return specs[specs.group.isin(groups)]["feature"].tolist()
