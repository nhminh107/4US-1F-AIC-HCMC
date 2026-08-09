"""Test ``app/ocr/bbox_utils.py`` (module_ocr.md mục 12.6).

Đối chiếu trực tiếp với ràng buộc CHECK của bảng ``OCR`` trong
``postgre_script.sql``: ``x_min < x_max``, ``y_min < y_max``, mọi giá trị
trong ``[0, 1]``.
"""

from __future__ import annotations

import pytest

from BackEnd.app.ocr.bbox_utils import (
    InvalidBBoxError,
    clip_pixel_bbox,
    pixel_to_normalized,
    validate_normalized_bbox,
)


class TestPixelToNormalized:
    def test_convert_dung_cong_thuc_x_chia_width_y_chia_height(self):
        x_min, x_max, y_min, y_max = pixel_to_normalized(
            10, 20, 110, 70, image_width=200, image_height=100
        )
        assert x_min == pytest.approx(10 / 200)
        assert x_max == pytest.approx(110 / 200)
        assert y_min == pytest.approx(20 / 100)
        assert y_max == pytest.approx(70 / 100)

    def test_bbox_hop_le_thoa_man_x_min_nho_hon_x_max(self):
        x_min, x_max, y_min, y_max = pixel_to_normalized(
            0, 0, 50, 50, image_width=100, image_height=100
        )
        assert x_min < x_max
        assert y_min < y_max

    def test_bbox_nam_ngoai_khung_anh_bi_clip_ve_0_1(self):
        # Bbox vượt hẳn ra ngoài mép phải/dưới ảnh -> clip về đúng biên ảnh
        # trước khi quy đổi, kết quả normalized phải nằm trong [0, 1].
        x_min, x_max, y_min, y_max = pixel_to_normalized(
            -50, -50, 500, 500, image_width=200, image_height=100
        )
        assert x_min == 0.0
        assert y_min == 0.0
        assert x_max == 1.0
        assert y_max == 1.0

    def test_bbox_hoan_toan_ngoai_khung_anh_raise_loi_ro_rang(self):
        # Cả bbox nằm hẳn bên phải ảnh -> sau khi clip, x_min == x_max == width
        # -> suy biến, không còn diện tích hợp lệ -> phải raise, không được
        # âm thầm lọt xuống thành OCRResult vi phạm CHECK constraint.
        with pytest.raises(InvalidBBoxError):
            pixel_to_normalized(300, 10, 400, 50, image_width=200, image_height=100)

    def test_kich_thuoc_anh_khong_hop_le_raise_loi(self):
        with pytest.raises(InvalidBBoxError):
            pixel_to_normalized(0, 0, 10, 10, image_width=0, image_height=100)


class TestClipPixelBBox:
    def test_gia_tri_am_duoc_clip_ve_0(self):
        x_min, y_min, x_max, y_max = clip_pixel_bbox(-10, -5, 50, 50, image_width=100, image_height=100)
        assert x_min == 0
        assert y_min == 0

    def test_gia_tri_vuot_bien_duoc_clip_ve_kich_thuoc_anh(self):
        x_min, y_min, x_max, y_max = clip_pixel_bbox(0, 0, 500, 500, image_width=100, image_height=80)
        assert x_max == 100
        assert y_max == 80


class TestValidateNormalizedBBox:
    def test_bbox_hop_le_khong_raise(self):
        validate_normalized_bbox(0.1, 0.5, 0.2, 0.6)

    def test_x_min_bang_x_max_raise_loi(self):
        with pytest.raises(InvalidBBoxError):
            validate_normalized_bbox(0.5, 0.5, 0.1, 0.6)

    def test_x_min_lon_hon_x_max_raise_loi(self):
        with pytest.raises(InvalidBBoxError):
            validate_normalized_bbox(0.6, 0.5, 0.1, 0.6)

    def test_y_min_lon_hon_bang_y_max_raise_loi(self):
        with pytest.raises(InvalidBBoxError):
            validate_normalized_bbox(0.1, 0.5, 0.6, 0.6)

    def test_gia_tri_ngoai_0_1_raise_loi(self):
        with pytest.raises(InvalidBBoxError):
            validate_normalized_bbox(-0.1, 0.5, 0.2, 0.6)
        with pytest.raises(InvalidBBoxError):
            validate_normalized_bbox(0.1, 1.5, 0.2, 0.6)
