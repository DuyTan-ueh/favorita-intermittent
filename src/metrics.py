"""Chỉ số đánh giá cho dự báo nhu cầu gián đoạn.

Chọn chỉ số nào là quyết định có hệ quả lớn với bài toán này, vì phần lớn chỉ
số quen thuộc đều hỏng khi dữ liệu có nhiều giá trị bằng không.

Những chỉ số KHÔNG dùng và lý do
--------------------------------
``sMAPE``
    Mẫu số chứa giá trị thực. Khi nhu cầu bằng không — chiếm khoảng một phần
    ba số quan sát ở đây — công thức không xác định. Nhiều thư viện âm thầm
    trả về 0 hoặc 200 thay vì báo lỗi, khiến kết quả trông vẫn hợp lý nhưng
    hoàn toàn vô nghĩa.

``MAPE``
    Cùng vấn đề, thậm chí nặng hơn vì không có giới hạn trên.

``MASE`` với mẫu số tính trên chuỗi thưa
    Mẫu số là sai số của dự báo ngây thơ trên tập huấn luyện. Với chuỗi mà
    nhiều đoạn dài toàn số không, mẫu số có thể bằng không hoặc rất gần
    không, làm chỉ số bùng nổ. Ở đây dùng RMSSE với xử lý mẫu số tường minh
    và báo cáo rõ số chuỗi bị loại.

Những chỉ số dùng
-----------------
``MAE``, ``RMSE``
    An toàn, dễ diễn giải theo đơn vị hàng hoá.

``WAPE``
    Mẫu số là tổng nhu cầu thực trên toàn bộ tập chứ không chia theo từng
    điểm, nên không hỏng khi gặp giá trị bằng không.

``RMSSE``
    Theo quy ước của cuộc thi M5, cho phép gộp kết quả giữa các chuỗi có quy
    mô khác nhau.

Ngoài ra, do khung Two-Stage tách bài toán làm hai, cần đánh giá riêng từng
giai đoạn thì mới biết sai số đến từ việc dự đoán sai NGÀY có đơn hay dự đoán
sai SỐ LƯỢNG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

KEYS = ["store_nbr", "item_nbr"]
EPS = 1e-9


# --------------------------------------------------------------------------- #
# Chỉ số cho dự báo cuối cùng
# --------------------------------------------------------------------------- #
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Sai số tuyệt đối có trọng số.

    Chia cho TỔNG nhu cầu thực chứ không chia từng điểm, nên an toàn với dữ
    liệu nhiều giá trị bằng không.
    """
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > EPS else np.nan


def scale_factor(y_train: np.ndarray, seasonality: int = 1) -> float:
    """Hệ số chuẩn hoá cho RMSSE: sai số bình phương của dự báo ngây thơ.

    Trả về ``nan`` khi chuỗi huấn luyện không đủ biến thiên để chuẩn hoá — ví
    dụ chuỗi toàn số không. Trả về ``nan`` thay vì một giá trị thay thế là có
    chủ ý: chuỗi đó phải bị loại khỏi phép tính tổng hợp và được đếm riêng,
    chứ không nên lặng lẽ nhận một con số tuỳ tiện.
    """
    if len(y_train) <= seasonality:
        return np.nan
    diff = y_train[seasonality:] - y_train[:-seasonality]
    val = float(np.mean(diff ** 2))
    return val if val > EPS else np.nan


def rmsse(y: np.ndarray, yhat: np.ndarray, scale: float) -> float:
    """Căn bậc hai của sai số bình phương đã chuẩn hoá, theo quy ước M5."""
    if not np.isfinite(scale) or scale <= EPS:
        return np.nan
    return float(np.sqrt(np.mean((y - yhat) ** 2) / scale))


# --------------------------------------------------------------------------- #
# Chỉ số riêng cho giai đoạn 1
# --------------------------------------------------------------------------- #
def occurrence_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> dict[str, float]:
    """Đánh giá khả năng nhận biết ngày CÓ phát sinh đơn.

    Dùng cả chỉ số phụ thuộc ngưỡng lẫn chỉ số độc lập với ngưỡng. PR-AUC
    được ưu tiên hơn ROC-AUC khi lớp dương hiếm, vì ROC-AUC có thể trông đẹp
    ngay cả khi mô hình gần như vô dụng trên lớp thiểu số.
    """
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Tổng hợp
# --------------------------------------------------------------------------- #
def evaluate_forecast(df: pd.DataFrame, scales: pd.DataFrame | None = None,
                      y_col: str = "y", pred_col: str = "yhat") -> dict:
    """Chỉ số tổng hợp trên toàn bộ tập kiểm tra.

    Báo cáo riêng phần ngày CÓ nhu cầu, vì đó là nơi sai số về số lượng thể
    hiện rõ. Nếu chỉ nhìn con số tổng, một mô hình luôn dự báo bằng không có
    thể trông khá tốt đơn giản vì phần lớn ngày đúng là bằng không.
    """
    y = df[y_col].to_numpy("float64")
    yhat = df[pred_col].to_numpy("float64")

    out = {
        "n_obs": int(len(df)),
        "zero_rate": float(np.mean(y == 0)),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "wape": wape(y, yhat),
    }

    pos = y > 0
    if pos.any():
        out["mae_positive"] = mae(y[pos], yhat[pos])
        out["rmse_positive"] = rmse(y[pos], yhat[pos])
        out["n_positive"] = int(pos.sum())

    if scales is not None:
        out.update(_aggregate_rmsse(df, scales, y_col, pred_col))
    return out


def _aggregate_rmsse(df: pd.DataFrame, scales: pd.DataFrame,
                     y_col: str, pred_col: str) -> dict:
    """RMSSE tính theo từng chuỗi rồi lấy trung bình.

    Chuỗi không có hệ số chuẩn hoá hợp lệ bị loại và đếm riêng, thay vì gán
    một giá trị thay thế làm sai lệch kết quả tổng hợp.
    """
    merged = df.merge(scales, on=KEYS, how="left")
    vals = []
    for _, g in merged.groupby(KEYS, sort=False):
        s = g["scale"].iloc[0]
        v = rmsse(g[y_col].to_numpy("float64"),
                  g[pred_col].to_numpy("float64"), s)
        if np.isfinite(v):
            vals.append(v)

    n_total = merged.groupby(KEYS, sort=False).ngroups
    return {
        "rmsse": float(np.mean(vals)) if vals else np.nan,
        "rmsse_n_series": len(vals),
        "rmsse_n_excluded": n_total - len(vals),
    }


def compute_scales(train: pd.DataFrame, y_col: str = "y",
                   seasonality: int = 1) -> pd.DataFrame:
    """Tính hệ số chuẩn hoá cho từng chuỗi từ dữ liệu HUẤN LUYỆN.

    Bắt buộc dùng dữ liệu huấn luyện: lấy từ tập kiểm tra sẽ khiến chỉ số
    phụ thuộc vào chính phần dữ liệu đang được đánh giá.
    """
    rows = []
    for key, g in train.sort_values("date").groupby(KEYS, sort=False):
        rows.append({"store_nbr": key[0], "item_nbr": key[1],
                     "scale": scale_factor(g[y_col].to_numpy("float64"),
                                           seasonality)})
    out = pd.DataFrame(rows)
    n_bad = out.scale.isna().sum()
    if n_bad:
        print(f"  [lưu ý] {n_bad}/{len(out)} chuỗi không có hệ số chuẩn hoá "
              f"hợp lệ (chuỗi huấn luyện không đủ biến thiên) — "
              f"sẽ loại khỏi RMSSE")
    return out


def evaluate_by_pattern(df: pd.DataFrame, series_meta: pd.DataFrame,
                        scales: pd.DataFrame | None = None,
                        y_col: str = "y",
                        pred_col: str = "yhat") -> pd.DataFrame:
    """Chỉ số tách theo nhóm mẫu nhu cầu — phục vụ trực tiếp RQ3.

    Đây là bảng trả lời câu hỏi trọng tâm của nghiên cứu: khung Two-Stage phát
    huy tác dụng đồng đều trên mọi mã hàng, hay chỉ ở nhóm bán thưa.
    """
    meta = series_meta[KEYS + ["pattern"]]
    merged = df.merge(meta, on=KEYS, how="left")

    rows = []
    for pattern, g in merged.groupby("pattern", sort=False):
        sub_scales = (scales[scales.set_index(KEYS).index
                             .isin(g.set_index(KEYS).index)]
                      if scales is not None else None)
        res = evaluate_forecast(g, sub_scales, y_col, pred_col)
        res["pattern"] = pattern
        res["n_series"] = g.groupby(KEYS, sort=False).ngroups
        rows.append(res)

    cols = ["pattern", "n_series", "n_obs", "zero_rate", "mae", "rmse",
            "wape", "mae_positive", "rmse_positive"]
    out = pd.DataFrame(rows)
    if "rmsse" in out.columns:
        cols.append("rmsse")
    return out[[c for c in cols if c in out.columns]].copy()
