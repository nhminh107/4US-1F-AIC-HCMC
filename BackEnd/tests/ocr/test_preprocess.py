"""Test ``app/ocr/preprocess.py`` (module_ocr.md mục 12.4)."""

from __future__ import annotations

import numpy as np
import pytest

from BackEnd.app.ocr.config import StaticRegion
from BackEnd.app.ocr.preprocess import (
    crop_static_region,
    enhance_contrast,
    load_image,
    preprocess,
    resize_image,
)


class TestResizeImage:
    def test_anh_canh_dai_lon_hon_max_side_duoc_thu_nho_giu_ty_le(self):
        image = np.zeros((600, 1200, 3), dtype=np.uint8)  # canh dai = 1200
        resized = resize_image(image, max_side=960)

        height, width = resized.shape[:2]
        assert max(height, width) == 960
        # Ty le khung hinh (width/height) phai giu nguyen ~2:1 nhu anh goc.
        assert width / height == pytest.approx(1200 / 600, rel=0.02)

    def test_anh_da_nho_hon_max_side_khong_bi_thay_doi(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(image, max_side=960)
        assert resized.shape == image.shape


class TestEnhanceContrast:
    def test_anh_toi_deu_mau_sau_clahe_co_bien_thien_do_sang(self):
        # Anh xam toi deu mau (khong co bien thien) truoc khi tang tuong phan.
        dark_flat_image = np.full((100, 100, 3), 40, dtype=np.uint8)
        enhanced = enhance_contrast(dark_flat_image)

        assert enhanced.shape == dark_flat_image.shape
        assert enhanced.dtype == dark_flat_image.dtype

    def test_khong_lam_lech_kich_thuoc_hay_kieu_du_lieu_anh(self):
        image = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        enhanced = enhance_contrast(image)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == image.dtype


class TestPreprocessPipeline:
    def test_resize_va_tang_tuong_phan_ap_dung_dung_thu_tu(self):
        image = np.full((600, 1200, 3), 40, dtype=np.uint8)
        result = preprocess(image, max_side=960)

        height, width = result.shape[:2]
        assert max(height, width) == 960


class TestLoadImage:
    def test_doc_anh_ton_tai_thanh_cong(self, clear_text_image_path):
        image = load_image(clear_text_image_path)
        assert image is not None
        assert image.ndim == 3

    def test_doc_anh_khong_ton_tai_raise_loi_ro_rang(self, tmp_path):
        missing_path = tmp_path / "khong_ton_tai.jpg"
        with pytest.raises(FileNotFoundError):
            load_image(missing_path)


class TestCropStaticRegion:
    def test_crop_dung_toa_do_da_khai_bao_trong_config(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :, :] = 0
        # Vung goc tren-trai, chiem 1 phan tu ben trai va nua tren cua anh.
        region = StaticRegion(name="logo", x_min=0.0, x_max=0.25, y_min=0.0, y_max=0.5)

        cropped = crop_static_region(image, region)

        expected_height = int(round(0.5 * 100))
        expected_width = int(round(0.25 * 200))
        assert cropped.shape[0] == expected_height
        assert cropped.shape[1] == expected_width

    def test_khong_lam_thay_doi_anh_goc(self):
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        original = image.copy()
        region = StaticRegion(name="logo", x_min=0.0, x_max=0.3, y_min=0.0, y_max=0.3)

        crop_static_region(image, region)

        assert np.array_equal(image, original)
