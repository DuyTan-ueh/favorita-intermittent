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
# Kiểm thử nhanh trên 200 chuỗi — chạy cái này trước
python -m src.pipeline --smoke

# Chạy đầy đủ
python -m src.pipeline --config config/default.yaml

# Chỉ tính thống kê chuỗi (bước 1)
python -m src.pipeline --stage stats

# Kiểm thử
pytest tests/ -v
```

Đổi thí nghiệm bằng cách copy `config/default.yaml`, sửa tham số, rồi chạy
với `--config`. Không sửa tham số trực tiếp trong code.

## Cấu trúc

```
config/default.yaml   Toàn bộ tham số thực nghiệm
src/
  config.py           Nạp + kiểm tra cấu hình, cố định seed
  data.py             Đọc dữ liệu, thống kê chuỗi, lấy mẫu phân tầng
  grid.py             Dựng lưới đầy đủ ngày × chuỗi
  features.py         Khai báo + sinh đặc trưng
  checks.py           Kiểm định rò rỉ, chia tập theo thời gian
  pipeline.py         Điều phối + CLI
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
  series_stats.parquet      Thống kê toàn bộ chuỗi
  series_selected.parquet   Tập chuỗi sau lọc và lấy mẫu
  feature_specs.csv         Bảng đặc trưng — dùng luôn cho Methodology
  features/part_*.parquet   Dữ liệu đã sinh đặc trưng
  metadata.json             Cấu hình, số liệu, định nghĩa fold
```

## Bước tiếp theo

Pipeline này dừng ở dữ liệu đã sẵn sàng huấn luyện. Phần còn lại:

- [ ] Baseline: Croston, SBA, TSB (`statsforecast`)
- [ ] Single-Stage: XGBoost regressor
- [ ] Two-Stage: XGBoost classifier + regressor
- [ ] Ablation cho RQ2, phân tầng cho RQ3
