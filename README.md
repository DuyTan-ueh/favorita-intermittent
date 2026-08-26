# Promotion-Aware Two-Stage ML for Intermittent Retail Demand Forecasting

Pipeline dữ liệu cho nghiên cứu dự báo nhu cầu gián đoạn trên bộ dữ liệu
Corporación Favorita.

## Câu hỏi nghiên cứu

| | Câu hỏi |
|---|---|
| **RQ1** | Two-Stage ML có tốt hơn dự báo một bước và Croston/SBA/TSB không? |
| **RQ2** | Đặc trưng khuyến mãi, lịch, sự kiện cải thiện dự báo bao nhiêu? |
| **RQ3** | Hiệu quả thay đổi thế nào giữa các nhóm mức độ gián đoạn khác nhau? |

## Cài đặt

```bash
pip install -r requirements.txt
```

Dữ liệu: tải từ [Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
(cần Join Competition + Accept Rules), giải nén vào `data/raw/`.
Trên Kaggle Notebook, pipeline tự dò đường dẫn.

> **Giấy phép dữ liệu:** Kaggle Competition Rules — được dùng cho nghiên cứu
> học thuật và phi thương mại, **không được phân phối lại dữ liệu gốc**.
> Vì vậy `data/` nằm trong `.gitignore`.

## Chạy

```bash
# 1. Kiểm thử nhanh trên 200 chuỗi — chạy cái này trước
python -m src.pipeline --smoke

# 2. Sinh dữ liệu đặc trưng (đầy đủ)
python -m src.pipeline --config config/default.yaml

# 3. Chạy thực nghiệm — thử nhanh trên fold cuối trước
python -m src.experiment --gap 0 --quick

# 4. Chạy đầy đủ — CHẠY TÁCH TỪNG GAP, mỗi lần một phiên
python -m src.experiment --gap 0
python -m src.experiment --gap 7

# Kiểm thử
pytest tests/ -v
```

Đổi thí nghiệm bằng cách copy `config/default.yaml`, sửa tham số, rồi chạy
với `--config`. Không sửa tham số trực tiếp trong code.

Nếu thiếu bộ nhớ ở bước thực nghiệm, dùng `--max-parts 3` để chỉ nạp một
phần các tệp lô. Cách này an toàn vì chuỗi đã được xáo trộn trước khi chia lô,
nên mỗi tệp là một mẫu đại diện.

### Thời gian chạy và checkpoint

Trên Kaggle, một giá trị gap với 3 fold và 5 bộ đặc trưng mất khoảng 5–8 giờ
tuỳ cấu hình máy được cấp. Chạy cả hai gap trong một phiên sẽ vượt giới hạn
12 giờ, nên **chạy tách từng gap**.

Kết quả được ghi lại sau mỗi fold. Nếu phiên bị ngắt giữa chừng, chạy lại
cùng lệnh sẽ bỏ qua các fold đã xong thay vì làm lại từ đầu. Muốn chạy lại
sạch, xoá thư mục `artifacts/<run>/results_gap*/checkpoints/`.

Giảm thời gian bằng cách hạ `experiment.n_estimators` hoặc bớt phần tử trong
`experiment.feature_sets`.

## Cấu trúc

```
config/default.yaml   Toàn bộ tham số thực nghiệm
src/
  config.py           Nạp + kiểm tra cấu hình, cố định seed
  data.py             Đọc dữ liệu, thống kê chuỗi, lấy mẫu phân tầng
  grid.py             Dựng lưới đầy đủ ngày × chuỗi
  features.py         Khai báo + sinh đặc trưng
  checks.py           Kiểm định rò rỉ, đồng nhất lô, chia tập theo thời gian
  metrics.py          Chỉ số đánh giá cho dữ liệu nhiều giá trị 0
  models.py           Croston/SBA/TSB, Single-Stage, Two-Stage
  experiment.py       Chạy thực nghiệm RQ1/RQ2/RQ3 + CLI
  pipeline.py         Điều phối bước dữ liệu + CLI
tests/                Kiểm thử, tập trung vào chống rò rỉ
artifacts/<run_name>/ Kết quả (không commit)
```

## Ba quyết định thiết kế đáng chú ý

### 1. Lưới phải dựng trước, và phải kiểm chứng

`train.csv` không lưu những ngày không bán được — nhu cầu bằng không tồn tại
dưới dạng *dòng bị khuyết*, không phải giá trị 0. Hệ quả nghiêm trọng hơn
việc thiếu dữ liệu: `shift(k)` trên lưới thủng sẽ lấy giá trị của một ngày
cách đó rất xa thay vì đúng k ngày trước, khiến đặc trưng lag mang ý nghĩa
khác hẳn tên gọi mà không hề báo lỗi.

`assert_grid_complete()` chạy sau mỗi lô để chặn tình huống này.

### 2. Đặc trưng tự khai báo tính khả dụng

Mỗi đặc trưng khai báo mình thuộc nhóm nào:

- `KNOWN_IN_ADVANCE` — lịch, ngày lễ, khuyến mãi. Được dùng giá trị của chính
  ngày đích, vì đây là thông tin đã có trong tay khi lập dự báo.
- `LAGGED` — dẫn xuất từ lịch sử nhu cầu. Bắt buộc trễ ít nhất `horizon` ngày.
- `STATIC` — thuộc tính không đổi theo thời gian.

`validate_specs()` đối chiếu khai báo với cấu hình và **dừng chương trình**
nếu có mâu thuẫn, thay vì để pipeline chạy xong mới phát hiện kết quả không
dùng được.

**Vì sao khuyến mãi được xếp vào nhóm biết trước:** cột `onpromotion` phản ánh
*kế hoạch* khuyến mãi, tức thông tin có sẵn trước khi bán hàng diễn ra. Điều
này khác hẳn việc suy ngược trạng thái khuyến mãi *từ* doanh số tăng đột biến
của chính ngày đó — cách làm ấy mới là rò rỉ, vì dùng kết quả để giải thích
nguyên nhân.

### 3. Kiểm định rò rỉ tự động

Khai báo cho biết *ý định*; kiểm định xác minh *thực tế*.

`truncation_invariance_test()` sinh đặc trưng hai lần — một lần trên dữ liệu
đầy đủ, một lần trên dữ liệu đã cắt bỏ phần tương lai — rồi so sánh các dòng
thuộc phần chung. Nếu một đặc trưng chỉ dùng thông tin quá khứ, hai kết quả
phải trùng khớp tuyệt đối.

Phép thử bắt được cả những lỗi mà đọc code khó thấy, ví dụ quên `shift` trước
khi tính cửa sổ trượt. Bộ kiểm thử có một trường hợp *cố tình* tạo rò rỉ để
chứng minh phép thử thực sự phát hiện được, chứ không phải luôn báo đạt.

### 4. Xáo trộn chuỗi trước khi chia lô

Thứ tự tự nhiên của dữ liệu phản ánh thời điểm chuỗi xuất hiện lần đầu. Nếu
giữ nguyên, lô đầu gom toàn chuỗi lịch sử dài (ít ngày không bán) còn lô cuối
toàn chuỗi ra mắt muộn. Lần chạy đầu tiên cho thấy chênh lệch tới 20 điểm phần
trăm về tỷ lệ ngày không bán giữa lô đầu và lô cuối.

Điều này nguy hiểm vì bước huấn luyện thường chỉ nạp một phần tệp để tiết kiệm
bộ nhớ — khi đó mẫu thu được thiên lệch mà không có dấu hiệu báo lỗi.
`check_batch_homogeneity()` giám sát và cảnh báo nếu tái diễn.

### 5. Hai kịch bản đánh giá song song

`gap_days` quyết định khoảng đệm giữa tập huấn luyện và tập kiểm tra, và mỗi
giá trị phản ánh một giả định vận hành khác nhau:

- `gap = 0` — ngầm định mô hình được huấn luyện lại mỗi ngày. Kịch bản lạc quan.
- `gap = 7` — huấn luyện một lần rồi dự báo trọn chu kỳ. Sát thực tế hơn.

Vì `gap_days` chỉ ảnh hưởng ranh giới fold chứ không ảnh hưởng đặc trưng, cả
hai bộ fold được sinh trong cùng một lần chạy thay vì chạy lại pipeline hai lần.

## Chỉ số đánh giá

Chọn chỉ số là quyết định có hệ quả với bài toán này vì nhiều chỉ số quen
thuộc hỏng khi dữ liệu có nhiều giá trị bằng không.

| Không dùng | Lý do |
|---|---|
| sMAPE, MAPE | Mẫu số chứa giá trị thực; không xác định khi nhu cầu bằng 0 |
| MASE (mặc định) | Mẫu số có thể bằng 0 trên chuỗi thưa, làm chỉ số bùng nổ |

| Dùng | Lý do |
|---|---|
| MAE, RMSE | An toàn, diễn giải theo đơn vị hàng hoá |
| WAPE | Mẫu số là tổng nhu cầu toàn tập, không chia từng điểm |
| RMSSE | Quy ước M5; chuỗi không có hệ số chuẩn hoá hợp lệ bị loại và đếm riêng |
| Precision/Recall/F1/PR-AUC | Đánh giá riêng giai đoạn 1 |

Chỉ số còn được báo cáo riêng cho những ngày CÓ nhu cầu. Nếu chỉ nhìn con số
tổng, một mô hình luôn dự báo bằng không có thể trông khá tốt đơn giản vì phần
lớn ngày đúng là bằng không.

## Cấu hình mặc định

| Tham số | Giá trị | Lý do |
|---|---|---|
| `start_date` | 2014-04-01 | `onpromotion` trống 100% trước mốc này |
| `horizon` | 7 ngày | |
| `lags` | 7, 14, 21, 28 | đều ≥ horizon, ràng buộc chống rò rỉ |
| `n_series` | 25.000 | 133k chuỗi vượt bộ nhớ; lấy mẫu phân tầng |
| `n_folds` | 3 | rolling-origin, không chia ngẫu nhiên |

## Kết quả sinh ra

```
artifacts/<run_name>/
  series_stats.parquet        Thống kê toàn bộ chuỗi
  series_selected.parquet     Tập chuỗi sau lọc và lấy mẫu
  feature_specs.csv           Bảng đặc trưng — dùng luôn cho Methodology
  batch_stats.csv             Kiểm tra đồng nhất giữa các lô
  features/part_*.parquet     Dữ liệu đã sinh đặc trưng
  folds_gap0.json             Định nghĩa fold, gap = 0
  folds_gap7.json             Định nghĩa fold, gap = 7
  metadata.json               Cấu hình, số liệu, định nghĩa fold
  results_gap{0,7}/
    metrics_by_fold.csv       Chỉ số từng fold × mô hình × bộ đặc trưng
    metrics_by_pattern.csv    Chỉ số theo nhóm mẫu nhu cầu (RQ3)
    summary.csv               Bảng tổng hợp (RQ1)
  gap_comparison.csv          So sánh gap = 0 và gap = 7
```

## Trạng thái

- [x] Dựng lưới đầy đủ + kiểm định toàn vẹn
- [x] Sinh đặc trưng + kiểm định chống rò rỉ
- [x] Baseline: Croston, SBA, TSB
- [x] Single-Stage: XGBoost hồi quy Tweedie
- [x] Two-Stage: XGBoost phân loại + hồi quy
- [x] Nghiên cứu loại trừ cho RQ2 (4 bộ đặc trưng)
- [x] Phân tầng theo nhóm nhu cầu cho RQ3
- [ ] Chạy trên dữ liệu Favorita thật
- [ ] Tinh chỉnh siêu tham số
- [ ] Kiểm chứng chéo trên M5
