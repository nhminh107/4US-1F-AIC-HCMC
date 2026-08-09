"""Tiền xử lý ảnh trước khi đưa vào ``OCREngine`` (module_ocr.md mục 5, 6, 8).

Gồm 3 việc độc lập nhau:
- Đọc ảnh từ đĩa (``load_image``).
- Resize + tăng tương phản (``preprocess``) — áp dụng cho TOÀN bộ ảnh trước
  khi gọi engine, giúp giảm thời gian xử lý (mục 8) và cải thiện khả năng đọc
  chữ mờ/tối.
- Cắt riêng 1 vùng ảnh theo toạ độ cố định đã khai báo trong ``config.py``
  (``crop_static_region``) — dùng cho ``dedup.py`` để tính pHash của đúng
  vùng logo/watermark, KHÔNG dùng để xoá/che vùng đó khỏi ảnh gốc (nhóm chủ
  định giữ lại text logo/watermark trong kết quả OCR cuối cùng).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from BackEnd.app.ocr import config
from BackEnd.app.ocr.config import StaticRegion


def load_image(image_path: str | Path) -> np.ndarray:
    """Đọc 1 ảnh từ đĩa bằng OpenCV (hệ màu BGR).

    Raise ``FileNotFoundError`` rõ ràng nếu không đọc được — ``cv2.imread``
    mặc định chỉ trả về ``None`` khi lỗi, dễ gây lỗi khó hiểu ở bước sau nếu
    không kiểm tra ngay tại đây.
    """

    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh tại: {path}")
    return image


def resize_image(image: np.ndarray, max_side: int = config.RESIZE_MAX_SIDE) -> np.ndarray:
    """Thu nhỏ ảnh sao cho cạnh dài nhất bằng ``max_side``, giữ nguyên tỉ lệ khung hình.

    Không phóng to ảnh vốn đã nhỏ hơn ``max_side`` — chỉ resize theo chiều
    giảm để không tạo thêm nhiễu nội suy không cần thiết.
    """

    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def enhance_contrast(
    image: np.ndarray,
    *,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Tăng tương phản bằng CLAHE, áp dụng riêng lên kênh độ sáng (L trong không gian LAB).

    Chỉ chỉnh kênh L (giữ nguyên 2 kênh màu a, b) để tránh làm lệch màu ảnh
    gốc — mục tiêu duy nhất là giúp chữ mờ/tối dễ đọc hơn cho engine, không
    phải chỉnh màu thẩm mỹ.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess(
    image: np.ndarray,
    *,
    max_side: int = config.RESIZE_MAX_SIDE,
) -> np.ndarray:
    """Áp dụng đầy đủ chuỗi tiền xử lý (resize -> tăng tương phản) lên 1 ảnh đã đọc sẵn.

    Nhận vào ``np.ndarray`` (không tự đọc file) để ``OCRExtractor`` kiểm soát
    rõ ràng thứ tự "đọc ảnh -> tiền xử lý" (module_ocr.md mục 5) mà không bị
    ẩn logic đọc file bên trong hàm tiền xử lý.
    """

    resized = resize_image(image, max_side=max_side)
    return enhance_contrast(resized)


def crop_static_region(image: np.ndarray, region: StaticRegion) -> np.ndarray:
    """Cắt đúng vùng ảnh tương ứng 1 ``StaticRegion`` (toạ độ normalized) khai báo trong config.

    Dùng bởi ``dedup.py`` để tính perceptual hash cho riêng vùng logo/watermark
    cố định — KHÔNG làm thay đổi ảnh gốc, chỉ trả về 1 bản crop mới.
    """

    height, width = image.shape[:2]

    x_min_px = int(round(region.x_min * width))
    x_max_px = int(round(region.x_max * width))
    y_min_px = int(round(region.y_min * height))
    y_max_px = int(round(region.y_max * height))

    # Giới hạn lại trong khung ảnh phòng trường hợp toạ độ khai báo lệch nhẹ
    # so với kích thước ảnh thực tế (vd config chung cho nhiều video có tỉ lệ
    # khung hình hơi khác nhau).
    x_min_px, x_max_px = max(0, x_min_px), min(width, x_max_px)
    y_min_px, y_max_px = max(0, y_min_px), min(height, y_max_px)

    return image[y_min_px:y_max_px, x_min_px:x_max_px]
