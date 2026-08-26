"""Nạp và kiểm tra cấu hình, đồng thời giải quyết đường dẫn dữ liệu.

Điểm quan trọng nhất của module này là hàm ``_validate``: nó chặn cấu hình
sai TỪ LÚC KHỞI ĐỘNG thay vì để pipeline chạy xong rồi mới phát hiện kết quả
bị rò rỉ thông tin tương lai.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Cấu hình không hợp lệ — dừng ngay, không chạy tiếp."""


# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #
def find_raw_dir() -> Path:
    """Tìm thư mục chứa dữ liệu thô, hoạt động cả trên Kaggle lẫn máy cá nhân.

    Kaggle đổi cấu trúc thư mục theo thời điểm tạo notebook nên phải dò đệ quy
    thay vì giả định một đường dẫn cố định.

    Nếu chỉ tìm thấy bản nén ``.7z`` — trường hợp mặc định khi vừa gắn dữ liệu
    cuộc thi vào một phiên làm việc mới — hàm sẽ tự giải nén sang thư mục ghi
    được. Thư mục ``/kaggle/input`` chỉ cho đọc nên không thể giải nén tại chỗ,
    và ``/kaggle/working`` bị xoá mỗi khi khởi động phiên mới, nên bước này cần
    chạy lại ở mỗi phiên. Tự động hoá để không phải nhớ.
    """
    csv_names = ["train.csv", "items.csv", "stores.csv", "holidays_events.csv"]

    search_roots = [Path("/kaggle/working/favorita_extracted"),
                    Path("data/raw"),
                    Path("/kaggle/input"),
                    Path("/kaggle/working")]

    # Ưu tiên tuyệt đối thư mục đã có train.csv giải nén sẵn
    for base in search_roots:
        if not base.exists():
            continue
        for root, _dirs, files in os.walk(base):
            if "train.csv" in files:
                return Path(root)

    # Không thấy bản giải nén -> tìm bản nén để tự xử lý
    archive_dir = None
    for base in search_roots:
        if not base.exists():
            continue
        for root, _dirs, files in os.walk(base):
            if "train.csv.7z" in files:
                archive_dir = Path(root)
                break
        if archive_dir:
            break

    if archive_dir is None:
        raise FileNotFoundError(
            "Không tìm thấy train.csv hoặc train.csv.7z.\n"
            "Trên Kaggle: Add Data cuộc thi favorita-grocery-sales-forecasting "
            "(nhớ Join Competition và Accept Rules trước).\n"
            "Ở máy cá nhân: đặt dữ liệu vào data/raw/."
        )

    return _extract_archives(archive_dir, csv_names)


def _extract_archives(src: Path, names: list[str]) -> Path:
    """Giải nén các tệp .7z cần dùng sang thư mục ghi được."""
    import shutil

    try:
        import py7zr
    except ImportError as exc:
        raise ImportError(
            "Cần thư viện py7zr để giải nén dữ liệu cuộc thi. "
            "Chạy: pip install py7zr"
        ) from exc

    dest = (Path("/kaggle/working/favorita_extracted")
            if Path("/kaggle/working").exists() else Path("data/raw"))
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Giải nén dữ liệu: {src} -> {dest}")
    for name in names:
        target = dest / name
        if target.exists():
            print(f"  {name:<24} đã có")
            continue
        plain, packed = src / name, src / f"{name}.7z"
        if plain.exists():
            shutil.copy(plain, target)
            print(f"  {name:<24} sao chép")
        elif packed.exists():
            with py7zr.SevenZipFile(packed, mode="r") as archive:
                archive.extractall(path=dest)
            print(f"  {name:<24} giải nén xong")
        else:
            raise FileNotFoundError(f"Không thấy {name} hoặc {name}.7z trong {src}")

    return dest


# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    raw: dict[str, Any]
    raw_dir: Path
    out_dir: Path

    # các nhóm tham số, ánh xạ thẳng từ YAML
    run: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    # ---- truy cập nhanh những giá trị dùng nhiều ----
    @property
    def horizon(self) -> int:
        return int(self.forecast["horizon"])

    @property
    def lags(self) -> list[int]:
        return list(self.features["lags"])

    @property
    def windows(self) -> list[int]:
        return list(self.features["rolling_windows"])

    @property
    def seed(self) -> int:
        return int(self.run["seed"])

    @property
    def batch_size(self) -> int:
        return int(self.output["batch_size"])


def load_config(path: str | Path = "config/default.yaml") -> Config:
    """Đọc YAML, kiểm tra tính hợp lệ, cố định seed, tạo thư mục kết quả."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file cấu hình: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    _validate(raw)

    out_dir = Path(raw["output"]["dir"]) / raw["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        raw=raw,
        raw_dir=find_raw_dir(),
        out_dir=out_dir,
        run=raw["run"],
        data=raw["data"],
        sampling=raw["sampling"],
        forecast=raw["forecast"],
        features=raw["features"],
        split=raw["split"],
        output=raw["output"],
    )

    _set_seed(cfg.seed)
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _validate(raw: dict) -> None:
    """Chặn những cấu hình gây rò rỉ thông tin tương lai hoặc vô nghĩa.

    Đây là hàng phòng thủ đầu tiên chống data leakage. Đặt kiểm tra ở đây
    nghĩa là sai cấu hình thì chương trình dừng ngay, thay vì chạy xong mới
    phát hiện toàn bộ kết quả không dùng được.
    """
    required = ["run", "data", "sampling", "forecast", "features",
                "split", "output"]
    for key in required:
        if key not in raw:
            raise ConfigError(f"Thiếu mục bắt buộc trong cấu hình: '{key}'")

    horizon = raw["forecast"]["horizon"]
    if horizon < 1:
        raise ConfigError("forecast.horizon phải >= 1")

    lags = raw["features"]["lags"]
    if not lags:
        raise ConfigError("features.lags không được rỗng")

    # ---- RÀNG BUỘC CỐT LÕI CHỐNG LEAKAGE ----
    # Dự báo ngày t+h tại thời điểm t: chỉ biết dữ liệu tới t.
    # Với hàng đích ở ngày D, lag_k lấy giá trị tại D-k.
    # Cần D-k <= D-h, tức k >= h. Lag nhỏ hơn horizon là dùng dữ liệu
    # chưa tồn tại tại thời điểm dự báo.
    bad = [k for k in lags if k < horizon]
    if bad:
        raise ConfigError(
            f"LEAKAGE: các lag {bad} nhỏ hơn horizon={horizon}. "
            f"Khi dự báo {horizon} ngày trước, giá trị của {horizon - 1} ngày "
            f"gần nhất CHƯA TỒN TẠI tại thời điểm dự báo. "
            f"Mọi lag phải >= {horizon}."
        )

    stats = raw["features"]["rolling_stats"]
    allowed = {"mean", "std", "min", "max", "sum", "median"}
    unknown = set(stats) - allowed
    if unknown:
        raise ConfigError(f"rolling_stats không hỗ trợ: {unknown}")

    if raw["split"]["n_folds"] < 1:
        raise ConfigError("split.n_folds phải >= 1")

    if raw["sampling"]["enabled"] and raw["sampling"]["n_series"] < 100:
        raise ConfigError("sampling.n_series quá nhỏ để phân tầng có ý nghĩa")
