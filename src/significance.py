"""Kiểm định thống kê cho việc so sánh độ chính xác dự báo.

Vì sao cần phần này
-------------------
Chênh lệch WAPE giữa các mô hình trong nghiên cứu này chỉ vào khoảng 0,005 đến
0,009. Một con số nhỏ như vậy, đứng một mình, không cho biết đó là khác biệt
thật hay chỉ là dao động ngẫu nhiên của mẫu. Phản biện sẽ hỏi ngay câu này, và
câu trả lời phải là một kiểm định chứ không phải một bảng số trung bình.

Ba lớp bằng chứng
-----------------
1. ``diebold_mariano`` — kiểm định chuẩn mực trong tài liệu dự báo, so sánh
   chuỗi chênh lệch hàm mất mát theo thời gian, có hiệu chỉnh tự tương quan.

2. ``paired_test_by_series`` — với hơn hai mươi nghìn chuỗi độc lập, kiểm định
   theo cặp trên từng chuỗi có lực mạnh hơn và ít giả định hơn. Báo cáo cả
   kiểm định t lẫn kiểm định dấu hạng Wilcoxon: cái sau không giả định phân
   phối chuẩn, phù hợp hơn với phân bố chênh lệch lệch mạnh.

3. ``holm_correction`` — khi so sánh nhiều cặp mô hình, xác suất có ít nhất một
   kết luận sai tăng nhanh theo số phép so sánh. Hiệu chỉnh Holm khắc phục
   điều này mà vẫn giữ được lực kiểm định tốt hơn Bonferroni thuần.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["store_nbr", "item_nbr"]


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray,
                    horizon: int = 1, power: int = 1,
                    small_sample_correction: bool = True) -> dict:
    """Kiểm định Diebold-Mariano so sánh độ chính xác của hai dự báo.

    Giả thuyết không là hai mô hình có độ chính xác kỳ vọng như nhau. Thống kê
    kiểm định dựa trên chuỗi chênh lệch hàm mất mát; phương sai được ước lượng
    bằng phương pháp nhất quán với tự tương quan, vì dự báo nhiều bước trước
    tạo ra sai số tương quan tới ``horizon - 1`` độ trễ.

    Áp dụng hiệu chỉnh mẫu nhỏ của Harvey, Leybourne và Newbold, vì thống kê
    gốc có xu hướng bác bỏ quá dễ khi mẫu ngắn.

    Parameters
    ----------
    power : int
        Bậc của hàm mất mát. ``1`` cho sai số tuyệt đối, phù hợp khi chỉ số
        đánh giá là WAPE; ``2`` cho sai số bình phương.

    Returns
    -------
    dict với thống kê kiểm định, giá trị p, và dấu hiệu mô hình nào tốt hơn.
    """
    d = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n,
                "better": "không đủ dữ liệu"}

    d_bar = float(np.mean(d))

    # Hai dự báo trùng khớp hoàn toàn: kết luận là không khác biệt, chứ không
    # phải lỗi tính toán. Tách riêng trường hợp này trước khi chia cho phương
    # sai, vì phương sai khi đó bằng không.
    if np.allclose(d, 0.0):
        return {"dm_stat": 0.0, "p_value": 1.0, "n": n,
                "mean_loss_diff": 0.0, "better": "không khác biệt"}

    # Phương sai dài hạn: cộng thêm các hiệp phương sai trễ vì sai số dự báo
    # nhiều bước trước có tương quan với nhau
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for lag in range(1, horizon):
        if lag >= n:
            break
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        var += 2 * cov

    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n,
                "better": "phương sai không hợp lệ"}

    dm = d_bar / np.sqrt(var / n)

    if small_sample_correction and horizon > 1:
        adj = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm *= adj

    df = max(n - 1, 1)
    p = float(2 * (1 - stats.t.cdf(abs(dm), df=df)))

    return {
        "dm_stat": float(dm),
        "p_value": p,
        "n": n,
        "mean_loss_diff": d_bar,
        "better": ("B" if d_bar > 0 else "A") if p < 0.05 else "không khác biệt",
    }


# --------------------------------------------------------------------------- #
# Kiểm định theo cặp trên từng chuỗi
# --------------------------------------------------------------------------- #
def paired_test_by_series(losses: pd.DataFrame, model_a: str, model_b: str,
                          loss_col: str = "mae") -> dict:
    """So sánh hai mô hình bằng kiểm định theo cặp trên từng chuỗi.

    Mỗi chuỗi đóng vai một quan sát ghép cặp: cùng một mặt hàng tại cùng một
    cửa hàng, được hai mô hình dự báo. Cách ghép cặp này loại bỏ phần lớn biến
    thiên giữa các chuỗi, nên phát hiện được cả những khác biệt nhỏ.

    Báo cáo song song kiểm định t và kiểm định Wilcoxon. Khi hai kiểm định cho
    cùng kết luận, độ tin cậy cao hơn hẳn; khi lệch nhau, thường là dấu hiệu
    phân bố chênh lệch có đuôi nặng và nên tin kiểm định Wilcoxon.
    """
    a = losses[losses.model_key == model_a].set_index(KEYS)[loss_col]
    b = losses[losses.model_key == model_b].set_index(KEYS)[loss_col]
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return {"n_series": len(common), "p_ttest": np.nan,
                "p_wilcoxon": np.nan, "verdict": "không đủ chuỗi"}

    x, y = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    diff = x - y                       # dương nghĩa là A tệ hơn
    ok = np.isfinite(diff)
    diff = diff[ok]

    t_stat, p_t = stats.ttest_rel(x[ok], y[ok])
    try:
        w_stat, p_w = stats.wilcoxon(diff)
    except ValueError:                 # mọi chênh lệch bằng nhau
        w_stat, p_w = np.nan, 1.0

    mean_diff = float(np.mean(diff))
    win_rate = float(np.mean(diff > 0))    # tỷ lệ chuỗi mà B tốt hơn

    # Với hơn hai mươi nghìn chuỗi, lực kiểm định cao tới mức gần như mọi
    # chênh lệch đều đạt ý nghĩa thống kê. Giá trị p khi đó không còn phân biệt
    # được đâu là khác biệt đáng kể trong thực tế, nên phải báo cáo kèm độ lớn
    # hiệu ứng. Quy ước thông dụng: 0,2 là nhỏ, 0,5 là vừa, 0,8 là lớn.
    sd = float(np.std(diff, ddof=1))
    if sd > 1e-12:
        cohen_d = mean_diff / sd
    elif abs(mean_diff) < 1e-12:
        cohen_d = 0.0          # hai mô hình trùng khớp hoàn toàn
    else:
        # Chênh lệch không đổi trên mọi chuỗi: hiệu ứng nhất quán tuyệt đối,
        # về mặt toán học là vô hạn. Trả về 0 ở đây sẽ nói ngược hoàn toàn ý
        # nghĩa, nên dùng vô cùng có dấu.
        cohen_d = float(np.inf) * np.sign(mean_diff)

    if abs(cohen_d) < 0.2:
        magnitude = "không đáng kể"
    elif abs(cohen_d) < 0.5:
        magnitude = "nhỏ"
    elif abs(cohen_d) < 0.8:
        magnitude = "vừa"
    else:
        magnitude = "lớn"

    if min(p_t, p_w) >= 0.05:
        verdict = "không khác biệt"
    else:
        verdict = f"{model_b} tốt hơn" if mean_diff > 0 else f"{model_a} tốt hơn"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_series": int(len(diff)),
        "mean_diff": mean_diff,
        "b_win_rate": win_rate,
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w),
        "cohen_d": float(cohen_d),
        "độ_lớn": magnitude,
        "verdict": verdict,
    }


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Hiệu chỉnh Holm-Bonferroni cho nhiều phép so sánh.

    So sánh nhiều cặp mô hình cùng lúc làm tăng xác suất có ít nhất một kết
    luận sai. Thủ tục Holm sắp xếp các giá trị p tăng dần rồi so từng giá trị
    với một ngưỡng nới lỏng dần, nên bảo toàn được sai lầm loại một trên toàn
    họ kiểm định mà vẫn mạnh hơn Bonferroni thuần.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * n
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (n - rank):
            reject[idx] = True
        else:
            break                       # dừng ngay khi gặp giá trị không đạt
    return reject


# --------------------------------------------------------------------------- #
# Điều phối
# --------------------------------------------------------------------------- #
def compare_by_pattern(losses: pd.DataFrame, series_meta: pd.DataFrame,
                       model_a: str, model_b: str,
                       loss_col: str = "mae") -> pd.DataFrame:
    """Kiểm định ghép cặp riêng trong TỪNG nhóm mẫu nhu cầu.

    Bảng phân tầng của câu hỏi nghiên cứu thứ ba chỉ đưa ra chênh lệch trung
    bình theo nhóm. Với biên độ vài phần nghìn, con số đó chưa đủ để khẳng
    định điều gì: cần biết chênh lệch có nhất quán trong nội bộ nhóm hay chỉ
    là dao động ngẫu nhiên.

    Hàm này thực hiện kiểm định ghép cặp riêng biệt trong từng nhóm, rồi hiệu
    chỉnh cho việc so sánh bốn nhóm cùng lúc. Kết quả trả lời được câu hỏi cụ
    thể mà phản biện sẽ đặt ra: kiến trúc hai giai đoạn thắng ở nhóm nào một
    cách đáng tin, và ở nhóm nào chênh lệch chỉ là nhiễu.
    """
    pooled = (losses.groupby(["model_key"] + KEYS, as_index=False)
              .agg(**{loss_col: (loss_col, "mean")}))
    meta = series_meta[KEYS + ["pattern"]].drop_duplicates()
    pooled = pooled.merge(meta, on=KEYS, how="left")

    rows = []
    for pattern, grp in pooled.groupby("pattern", sort=False):
        if pd.isna(pattern):
            continue
        res = paired_test_by_series(grp, model_a, model_b, loss_col)
        res["pattern"] = pattern
        rows.append(res)

    out = pd.DataFrame(rows)
    if not len(out):
        return out

    # Bốn nhóm được kiểm định cùng lúc nên xác suất có ít nhất một kết luận
    # sai tăng lên; hiệu chỉnh Holm đưa nó về mức kiểm soát được.
    out["reject_holm"] = holm_correction(out.p_wilcoxon.fillna(1.0).tolist())
    out["ý_nghĩa"] = np.where(out.reject_holm, "có", "không")

    order = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    out["_o"] = out.pattern.map({p: i for i, p in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Độ ổn định qua nhiều seed
# --------------------------------------------------------------------------- #
# Quy tắc quyết định được khai báo Ở ĐÂY, trước khi biết kết quả thật. Mục đích
# là ngăn việc diễn giải "vững" hay "chỉ là gợi ý" bị chọn sau khi đã nhìn thấy
# số liệu — một dạng thiên lệch xác nhận rất dễ mắc phải khi tự đánh giá công
# trình của chính mình.
#
# Quy tắc tính theo TỶ LỆ seed đồng thuận, không theo số tuyệt đối — vì 3/3 seed
# đồng thuận là bằng chứng mạnh hơn 3/5, dù cùng "ba seed thắng". Số seed tối
# thiểu để một tỷ lệ có ý nghĩa là 3; dưới mức đó không đủ để kết luận gì.
def _seed_verdict(consistent: int, n: int) -> str:
    if n < 3:
        return "không đủ seed để kết luận (cần tối thiểu 3)"
    frac = consistent / n
    if frac == 1.0:
        return "vững — trình bày như phát hiện chính của bài"
    if frac >= 0.8:
        return "gợi ý mạnh — diễn đạt thận trọng, nêu rõ seed đảo chiều"
    if frac >= 0.6:
        return "gợi ý yếu — chỉ nêu trong Discussion, không đưa vào Kết luận"
    return "không đủ bằng chứng — coi chiều thắng là ngẫu nhiên"


def seed_stability_table(results: pd.DataFrame, model_a: str, model_b: str,
                         feature_set: str = "full",
                         metric: str = "wape") -> pd.DataFrame:
    """Bảng chi tiết từng seed, không chỉ trung bình.

    Chỉ nhìn trung bình và độ lệch chuẩn có thể che giấu một seed lệch hẳn: bốn
    seed gần nhau cộng một seed ngoại lệ vẫn cho ra độ lệch chuẩn trông chấp
    nhận được. Bảng liệt kê từng seed để phát hiện trường hợp đó, và để người
    đọc tự kiểm tra thay vì tin vào một con số tổng hợp duy nhất.

    Cột quan trọng nhất là ``delta``: đây là đại lượng cần so sánh với độ lệch
    chuẩn giữa các seed để trả lời câu hỏi trung tâm — chênh lệch giữa hai mô
    hình có vượt qua nhiễu ngẫu nhiên của quá trình huấn luyện hay không.
    """
    sub = results[results.feature_set == feature_set]
    a = sub[sub.model == model_a].groupby("seed")[metric].mean()
    b = sub[sub.model == model_b].groupby("seed")[metric].mean()

    out = pd.DataFrame({model_a: a, model_b: b}).dropna()
    if not len(out):
        return out

    out["delta"] = out[model_a] - out[model_b]
    out[f"{model_b}_thắng"] = out["delta"] > 0
    return out.reset_index()


def summarise_seed_stability(table: pd.DataFrame, model_a: str,
                             model_b: str) -> dict:
    """Tóm tắt bảng độ ổn định thành một quyết định theo quy tắc đã khai báo.

    Trả về cả số liệu lẫn nhãn quyết định, để in ra cùng lúc và không thể tách
    số liệu khỏi diễn giải — tránh tình huống chọn cách diễn đạt sau khi đã
    thấy con số.
    """
    if not len(table):
        return {"n_seeds": 0, "n_wins": 0, "verdict": "không có dữ liệu"}

    win_col = f"{model_b}_thắng"
    n = len(table)
    wins = int(table[win_col].sum())
    delta_mean = float(table.delta.mean())
    delta_std = float(table.delta.std(ddof=1)) if n > 1 else float("nan")
    ratio = (abs(delta_mean) / delta_std) if delta_std and delta_std > 1e-9 \
        else float("inf")

    # Quy tắc áp cho phía có nhiều seed thắng hơn — 5/5 và 0/5 cùng mức "vững",
    # chỉ khác chiều thắng.
    consistent_side = max(wins, n - wins)

    return {
        "n_seeds": n,
        "n_wins_b": wins,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_over_std": ratio,
        "verdict": _seed_verdict(consistent_side, n),
    }


def per_series_losses(pred: pd.DataFrame, model_key: str,
                      fold: int) -> pd.DataFrame:
    """Tổng hợp sai số theo từng chuỗi, phục vụ kiểm định ghép cặp.

    Lưu ở mức chuỗi thay vì mức quan sát giữ cho tệp kết quả nhỏ gọn mà vẫn đủ
    thông tin cho mọi kiểm định ghép cặp về sau.
    """
    pred = pred.copy()
    pred["abs_err"] = (pred.y - pred.yhat).abs()
    pred["sq_err"] = (pred.y - pred.yhat) ** 2

    out = (pred.groupby(KEYS, as_index=False)
           .agg(mae=("abs_err", "mean"), mse=("sq_err", "mean"),
                sum_abs=("abs_err", "sum"), sum_y=("y", "sum"),
                n_obs=("y", "size")))
    out["model_key"] = model_key
    out["fold"] = fold
    return out


def compare_all(losses: pd.DataFrame, reference: str,
                loss_col: str = "mae") -> pd.DataFrame:
    """So sánh mọi mô hình với một mô hình tham chiếu.

    Gộp các fold lại trước khi kiểm định: mỗi chuỗi khi đó có một giá trị sai
    số trung bình duy nhất, và các chuỗi có thể coi là độc lập với nhau.
    """
    pooled = (losses.groupby(["model_key"] + KEYS, as_index=False)
              .agg(**{loss_col: (loss_col, "mean")}))

    models = [m for m in pooled.model_key.unique() if m != reference]
    rows = [paired_test_by_series(pooled, reference, m, loss_col)
            for m in models]
    out = pd.DataFrame(rows)
    if not len(out):
        return out

    out["reject_holm"] = holm_correction(out.p_wilcoxon.fillna(1.0).tolist())
    out["ý_nghĩa"] = np.where(out.reject_holm, "có", "không")
    return out.sort_values("mean_diff", ascending=False)
