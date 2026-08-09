"""Interface trừu tượng cho engine OCR (module_ocr.md mục 5).

``OCREngine`` là ranh giới kiến trúc duy nhất giữa ``OCRExtractor`` và model
cụ thể đang dùng để detect + recognize chữ. Hiện tại chỉ có 1 class engine
được triển khai (``MonkeyOCREngine``, mục 4.3) — nhưng giữ interface này tách
riêng để nếu sau này cần đổi/thêm model, chỉ cần viết thêm 1 class engine mới
tuân theo đúng ``Protocol`` này, không phải sửa ``OCRExtractor`` hay contract
``OCRResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class RawTextRegion:
    """1 vùng chữ do engine nhận dạng được — TRƯỚC khi chuẩn hoá thành ``OCRResult``.

    Khác với ``OCRResult`` (data contract, ``pipeline.py``):
    - ``bbox`` ở đây là toạ độ PIXEL trên ảnh đã tiền xử lý (đã resize), chưa
      quy đổi normalized ``[0, 1]`` — việc quy đổi thuộc trách nhiệm của
      ``OCRExtractor`` (qua ``bbox_utils.py``), không phải của engine.
    - ``text`` chưa được hậu xử lý (chưa strip khoảng trắng thừa, chưa lọc
      theo confidence) — việc đó thuộc ``postprocess.py``.
    - Có thêm ``confidence`` (engine OCR luôn có điểm tin cậy, khác Caption -
      xem ghi chú ở ``CaptionResult.confidence``) để ``postprocess.py`` lọc
      vùng chữ chất lượng thấp trước khi trả về ``OCRExtractor``.
    """

    text: str
    bbox: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) pixel
    confidence: float | None = None
    language: str | None = None


class OCREngine(Protocol):
    """Interface tối thiểu 1 engine OCR phải cài đặt để cắm vào ``OCRExtractor``."""

    def run(self, image: np.ndarray) -> list[RawTextRegion]:
        """Chạy detect + recognize trên 1 ảnh (numpy array, BGR, đã tiền xử lý).

        Trả về danh sách rỗng nếu ảnh không có chữ — đây là kết quả HỢP LỆ,
        không phải lỗi (module_ocr.md mục 3).
        """
        ...
