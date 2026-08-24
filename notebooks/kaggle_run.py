# =============================================================================
# KAGGLE RUNNER
# Chạy pipeline trên Kaggle Notebook mà không cần upload repo.
#
# CÁCH DÙNG
#   1. Add Data -> Competitions -> favorita-grocery-sales-forecasting
#      (Join Competition + Accept Rules trước)
#   2. Paste từng CELL bên dưới
#
# Vì sao cần file này: Kaggle Notebook không clone git repo trực tiếp được,
# nên các module được ghi ra đĩa bằng %%writefile rồi import như bình thường.
# Cách này giữ code trên GitHub là nguồn duy nhất, notebook chỉ là lớp vỏ chạy.
# =============================================================================


# %% ==========================================================================
# CELL 1 -- Giải nén dữ liệu (.7z -> .csv)
#           /kaggle/input chỉ đọc nên phải ghi ra /kaggle/working
# =============================================================================
CELL_1 = r'''
import os, subprocess, sys, shutil
from time import time

try:
    import py7zr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"],
                   check=True)
    import py7zr

RAW = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "favorita" in root.lower() and any(
            f.startswith("train.csv") for f in files):
        RAW = root
        break
assert RAW, "Chưa Add Data cuộc thi favorita-grocery-sales-forecasting"

DEST = "/kaggle/working/favorita_extracted"
os.makedirs(DEST, exist_ok=True)

for name in ["train.csv", "items.csv", "stores.csv", "holidays_events.csv"]:
    if os.path.exists(f"{DEST}/{name}"):
        print(f"  {name:<24} đã có")
        continue
    if os.path.exists(f"{RAW}/{name}"):
        shutil.copy(f"{RAW}/{name}", f"{DEST}/{name}")
        print(f"  {name:<24} copy")
    else:
        t0 = time()
        with py7zr.SevenZipFile(f"{RAW}/{name}.7z", mode="r") as z:
            z.extractall(path=DEST)
        print(f"  {name:<24} giải nén {time()-t0:.0f}s")

print("\nDữ liệu sẵn sàng tại", DEST)
'''


# %% ==========================================================================
# CELL 2 -- Ghi các module ra đĩa
#           Copy nội dung từng file trong src/ của repo vào đây.
#           Ví dụ với config.py:
# =============================================================================
CELL_2 = r'''
import os
os.makedirs("/kaggle/working/src", exist_ok=True)
os.makedirs("/kaggle/working/config", exist_ok=True)
open("/kaggle/working/src/__init__.py", "w").close()

# --- Với mỗi module, dùng %%writefile ở đầu một cell riêng: ---
#
#   %%writefile /kaggle/working/src/config.py
#   <dán toàn bộ nội dung src/config.py>
#
#   %%writefile /kaggle/working/src/data.py
#   <dán toàn bộ nội dung src/data.py>
#
#   ... tương tự cho grid.py, features.py, checks.py, pipeline.py
#   ... và config/default.yaml
#
# Cách nhanh hơn: nén repo thành .zip, upload lên Kaggle như một Dataset,
# rồi giải nén vào /kaggle/working. Khi đó chỉ cần:
#
#   !cp -r /kaggle/input/<ten-dataset>/favorita-intermittent/* /kaggle/working/
'''


# %% ==========================================================================
# CELL 3 -- Chạy pipeline
# =============================================================================
CELL_3 = r'''
import os, sys
os.chdir("/kaggle/working")
sys.path.insert(0, "/kaggle/working")

# Chạy thử trên tập nhỏ trước — bắt lỗi sớm, tiết kiệm thời gian
!python -m src.pipeline --smoke
'''


# %% ==========================================================================
# CELL 4 -- Chạy đầy đủ (sau khi smoke test đạt)
# =============================================================================
CELL_4 = r'''
!python -m src.pipeline --config config/default.yaml
'''


# %% ==========================================================================
# CELL 5 -- Kiểm tra kết quả
# =============================================================================
CELL_5 = r'''
import json, glob
import pandas as pd

RUN = "artifacts/baseline_h7"

with open(f"{RUN}/metadata.json", encoding="utf-8") as fh:
    meta = json.load(fh)
print(f"Chuỗi     : {meta['n_series']:,}")
print(f"Dòng      : {meta['n_rows']:,}")
print(f"Đặc trưng : {meta['n_features']}")
print(f"Horizon   : {meta['horizon']} ngày")

specs = pd.read_csv(f"{RUN}/feature_specs.csv")
print("\nĐặc trưng theo nhóm:")
print(specs.groupby(["group", "availability"]).size().to_string())

parts = sorted(glob.glob(f"{RUN}/features/part_*.parquet"))
df = pd.read_parquet(parts[0])
print(f"\nLô đầu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f"Tỷ lệ ngày không bán: {(df.y == 0).mean()*100:.1f}%")

sel = pd.read_parquet(f"{RUN}/series_selected.parquet")
print("\nPhân nhóm mẫu nhu cầu:")
print(sel.pattern.value_counts().to_string())
'''
