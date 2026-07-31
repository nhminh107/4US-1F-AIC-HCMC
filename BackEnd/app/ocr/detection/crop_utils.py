"""
Cat anh theo bbox da detect duoc.

Truoc day module nay con ghi anh crop ra temp_crops/, nhung voi hon 2000 keyframe
trong Test_data thi moi keyframe cho ra 3-4 vung logo/watermark -> hon 15000 file anh
(xem muc "phat sinh nhieu file anh crop" trong Remake_module_ocr.md). Giai phap: bo
han viec ghi file tam, chi cat anh va giu lai duoi dang numpy array trong RAM, roi
truyen thang cho Recognition trong cung 1 tien trinh Python (xem run_ocr_pipeline.py).
"""
from typing import List

import numpy as np


def crop_region(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    # Tach 4 toa do cua bbox.
    x_min, y_min, x_max, y_max = bbox
    # Lay chieu cao, chieu rong cua anh goc.
    h, w = image.shape[:2]
    # Gioi han toa do min khong nho hon 0.
    x_min, y_min = max(0, x_min), max(0, y_min)
    # Gioi han toa do max khong vuot qua kich thuoc anh.
    x_max, y_max = min(w, x_max), min(h, y_max)
    # Cat (slice) vung anh nam trong bbox va tra ve.
    return image[y_min:y_max, x_min:x_max]
