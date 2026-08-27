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
def resolve_device(requested: str = "auto") -> str:
    """Chọn thiết bị tính toán, tự lùi về CPU nếu không có GPU.

    XGBoost từ phiên bản 2.0 dùng tham số ``device`` thay cho
    ``tree_method="gpu_hist"`` đã bị loại bỏ. Cách gọi đúng hiện nay là đặt
    ``device="cuda"`` cùng với ``tree_method="hist"``.
    """
    if requested == "cpu":
        return "cpu"

    try:
        import subprocess
        subprocess.run(["nvidia-smi"], capture_output=True, check=True,
                       timeout=10)
        has_gpu = True
    except Exception:
        has_gpu = False

    if requested == "cuda" and not has_gpu:
        print("  [cảnh báo] yêu cầu GPU nhưng không phát hiện được, dùng CPU")
        return "cpu"
    return "cuda" if has_gpu else "cpu"


def _xgb_params(objective: str, seed: int, device: str = "cpu") -> dict:
    base = {
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",     # bắt buộc dùng cùng device="cuda"
        "device": device,
        "random_state": seed,
        "n_jobs": -1,
    }
    base["objective"] = objective
    return base


# --------------------------------------------------------------------------- #
# Hàm mất mát cho giai đoạn hai
# --------------------------------------------------------------------------- #
# Giai đoạn hai chỉ học trên các quan sát có nhu cầu dương, nên phân phối mục
# tiêu bị chặn dưới và lệch phải mạnh. Sai số bình phương — lựa chọn mặc định
# quen thuộc — giả định sai số đối xứng trên toàn trục thực, nên không phù hợp
# và còn bị các đợt tăng vọt hiếm gặp chi phối.
#
# Tài liệu về mô hình hurdle đề xuất vài hướng thay thế, mỗi hướng ứng với một
# giả định khác nhau về phân phối nhu cầu dương:
#
#   gamma
#       Phân phối liên tục, dương, lệch phải. Đây là lựa chọn của khung
#       Zero-Inflated Gamma trong tài liệu về phụ tùng thay thế. Phù hợp với
#       dữ liệu Favorita vì có nhóm hàng cân theo ký nên nhu cầu không thuần
#       số đếm.
#
#   poisson
#       Phân phối đếm. Giả định phương sai bằng kỳ vọng, thường bị vi phạm khi
#       nhu cầu phân tán quá mức.
#
#   log_squared
#       Biến đổi log rồi dùng sai số bình phương, hoàn nguyên khi dự báo. Cách
#       làm thực dụng, được dùng trong một số khung Two-Stage đã công bố.
#
#   absolute
#       Sai số tuyệt đối. Đáng chú ý vì WAPE — chỉ số đánh giá chính của
#       nghiên cứu này — cũng dựa trên sai số tuyệt đối. Huấn luyện bằng sai số
#       bình phương trong khi đánh giá bằng sai số tuyệt đối là một sự lệch
#       mục tiêu; tuỳ chọn này loại bỏ sự lệch đó. Lưu ý đánh đổi: sai số tuyệt
#       đối cho trung vị có điều kiện chứ không phải kỳ vọng, nên tích
#       ``P(y>0) x du_bao`` không còn là kỳ vọng của nhu cầu.
#
#   squared
#       Giữ lại làm mốc so sánh, tương ứng cấu hình ban đầu.
#
# Tham số ``shift``: nhu cầu dương bắt đầu từ 1 chứ không phải 0, nên có thể mô
# hình hoá phần dư ``y - 1`` rồi cộng lại khi dự báo. Đây là cách làm của phân
# phối nhị thức âm dịch một đơn vị trong tài liệu.

STAGE2_OBJECTIVES = {
    "squared": {"objective": "reg:squarederror", "transform": None},
    "gamma": {"objective": "reg:gamma", "transform": None},
    "poisson": {"objective": "count:poisson", "transform": None},
    "absolute": {"objective": "reg:absoluteerror", "transform": None},
    "log_squared": {"objective": "reg:squarederror", "transform": "log1p"},
}


def _fit_stage2(train_pos: pd.DataFrame, feature_cols: list[str],
                spec: dict, seed: int, device: str, n_estimators: int,
                shift: bool):
    """Huấn luyện mô hình độ lớn nhu cầu theo cấu hình hàm mất mát."""
    import xgboost as xgb

    y = train_pos["y"].to_numpy("float64")
    if shift:
        y = y - 1.0                    # nhu cầu dương nhỏ nhất là 1
    if spec["transform"] == "log1p":
        y = np.log1p(y)
    elif spec["objective"] in ("reg:gamma", "count:poisson"):
        # hai phân phối này yêu cầu mục tiêu dương ngặt
        y = np.clip(y, 1e-6, None)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        **_xgb_params(spec["objective"], seed, device))
    model.fit(train_pos[feature_cols], y, verbose=False)
    return model


def _predict_stage2(model, X: pd.DataFrame, spec: dict,
                    shift: bool) -> np.ndarray:
    """Dự báo độ lớn và hoàn nguyên mọi phép biến đổi đã áp dụng."""
    pred = model.predict(X).astype("float64")
    if spec["transform"] == "log1p":
        pred = np.expm1(pred)
    if shift:
        pred = pred + 1.0
    return np.clip(pred, 0, None)


def fit_single_stage(train: pd.DataFrame, test: pd.DataFrame,
                     feature_cols: list[str], seed: int = 42,
                     n_estimators: int = 300, device: str = "cpu",
                     **_) -> tuple:
    """Hồi quy trực tiếp số lượng, không tách bài toán.

    Dùng hàm mất mát Tweedie thay vì bình phương sai số: phân phối Tweedie có
    khối xác suất tại không cộng với phần liên tục dương, đúng dạng dữ liệu
    nhu cầu gián đoạn. Dùng bình phương sai số ở đây sẽ tạo mốc so sánh yếu
    một cách không cần thiết.
    """
    import xgboost as xgb

    params = _xgb_params("reg:tweedie", seed, device)
    params["tweedie_variance_power"] = 1.3

    model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
    model.fit(train[feature_cols], train["y"], verbose=False)

    out = test[KEYS + ["date", "y"]].copy()
    out["yhat"] = np.clip(model.predict(test[feature_cols]), 0, None)
    return out, model


def fit_two_stage(train: pd.DataFrame, test: pd.DataFrame,
                  feature_cols: list[str], seed: int = 42,
                  n_estimators: int = 300, device: str = "cpu",
                  stage2: str = "squared", shift: bool = False) -> tuple:
    """Khung hai giai đoạn: phân loại khả năng có đơn, rồi hồi quy số lượng.

    Giai đoạn hai CHỈ được huấn luyện trên các quan sát có nhu cầu dương. Đây
    là điểm mấu chốt của phương pháp: mô hình số lượng không bị hàng loạt giá
    trị bằng không kéo về không, nên học được phân phối số lượng thật sự.

    Dự báo cuối là tích của xác suất và độ lớn dự báo.
    """
    import xgboost as xgb

    spec = STAGE2_OBJECTIVES.get(stage2)
    if spec is None:
        raise ValueError(f"stage2 không hợp lệ: {stage2}. "
                         f"Chọn một trong {list(STAGE2_OBJECTIVES)}")

    # --- Giai đoạn 1: có phát sinh đơn hay không ---
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        eval_metric="logloss",
        **_xgb_params("binary:logistic", seed, device))
    clf.fit(train[feature_cols], train["y_occurrence"], verbose=False)

    # --- Giai đoạn 2: số lượng, chỉ trên ngày có đơn ---
    pos = train[train.y_occurrence == 1]
    reg = _fit_stage2(pos, feature_cols, spec, seed, device,
                      n_estimators, shift)

    prob = clf.predict_proba(test[feature_cols])[:, 1]
    magnitude = _predict_stage2(reg, test[feature_cols], spec, shift)

    out = test[KEYS + ["date", "y", "y_occurrence"]].copy()
    out["prob_occurrence"] = prob
    out["pred_magnitude"] = magnitude
    out["yhat"] = prob * magnitude
    return out, (clf, reg)


# --------------------------------------------------------------------------- #
# Nhóm đặc trưng cho nghiên cứu loại trừ (RQ2)
# --------------------------------------------------------------------------- #
FEATURE_SETS = {
    "historical": ["lag", "rolling", "intermittency"],
    "hist_calendar": ["lag", "rolling", "intermittency", "calendar"],
    "hist_cal_holiday": ["lag", "rolling", "intermittency", "calendar",
                         "holiday"],
    "hist_cal_hol_promo": ["lag", "rolling", "intermittency", "calendar",
                           "holiday", "promotion"],
    "full": ["lag", "rolling", "intermittency", "calendar", "holiday",
             "promotion", "static"],
}


def select_features(specs: pd.DataFrame, groups: list[str]) -> list[str]:
    """Chọn cột đặc trưng theo nhóm, phục vụ nghiên cứu loại trừ.

    Năm cấu hình được xếp chồng dần, mỗi bước thêm đúng MỘT nhóm đặc trưng.
    Nhờ vậy chênh lệch giữa hai bước liền kề chính là đóng góp riêng của nhóm
    vừa thêm vào:

      historical -> hist_calendar          : đóng góp của đặc trưng lịch
      hist_calendar -> hist_cal_holiday    : đóng góp của ngày lễ, sự kiện
      hist_cal_holiday -> hist_cal_hol_promo : đóng góp của KHUYẾN MÃI
      hist_cal_hol_promo -> full           : đóng góp của thuộc tính tĩnh

    Bước áp chót là bước quan trọng nhất, vì nó trả lời trực tiếp RQ2. Nếu bỏ
    bước này và nhảy thẳng từ ``hist_cal_holiday`` sang ``full``, đóng góp của
    khuyến mãi sẽ bị lẫn với đóng góp của thuộc tính tĩnh và không tách được.
    """
    return specs[specs.group.isin(groups)]["feature"].tolist()
