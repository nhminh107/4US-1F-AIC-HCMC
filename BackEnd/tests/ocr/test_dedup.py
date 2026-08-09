"""Test ``app/ocr/dedup.py`` (module_ocr.md mục 12.5).

Test case quan trọng nhất của module dedup: banner/ticker chạy chữ (layout cố
định, nội dung đổi liên tục) KHÔNG được coi là trùng lặp — đây chính là lỗi
đã fix so với bản thiết kế cũ (module_ocr.md mục 7).
"""

from __future__ import annotations

import numpy as np
import pytest

from BackEnd.app.ocr.config import StaticRegion
from BackEnd.app.ocr.dedup import (
    StaticRegionDedupCache,
    apply_static_region_dedup,
    hamming_distance,
    is_duplicate_hash,
    perceptual_hash,
)
from BackEnd.app.ocr.engines.base import RawTextRegion

LOGO_REGION = StaticRegion(name="logo", x_min=0.0, x_max=0.3, y_min=0.0, y_max=0.3)


def _solid_image(color: tuple[int, int, int], size: tuple[int, int] = (200, 200)) -> np.ndarray:
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _checkerboard_image(*, invert: bool = False, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Ảnh caro đen/trắng - dùng để test aHash vì ảnh MỘT màu đồng nhất luôn
    cho hash bằng nhau (mọi điểm ảnh đều bằng chính giá trị trung bình của
    ảnh), không phản ánh được trường hợp 2 ảnh THỰC SỰ khác nội dung.
    """

    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    half_h, half_w = height // 2, width // 2
    light, dark = (255, 255, 255), (0, 0, 0)
    if invert:
        light, dark = dark, light
    image[:half_h, :half_w] = light
    image[:half_h, half_w:] = dark
    image[half_h:, :half_w] = dark
    image[half_h:, half_w:] = light
    return image


class TestPerceptualHashAndHamming:
    def test_hamming_distance_giua_2_hash_giong_het_la_0(self):
        image = _solid_image((10, 20, 30))
        hash_a = perceptual_hash(image)
        hash_b = perceptual_hash(image.copy())
        assert hamming_distance(hash_a, hash_b) == 0

    def test_2_anh_khac_han_nhau_khong_duoc_coi_la_trung(self):
        checkerboard = _checkerboard_image()
        inverted_checkerboard = _checkerboard_image(invert=True)
        assert not is_duplicate_hash(
            perceptual_hash(checkerboard), perceptual_hash(inverted_checkerboard)
        )

    def test_vuot_nguong_hamming_khong_coi_la_trung(self):
        # Ep 2 hash lech nhau nhieu bit hon nguong mac dinh (4) bang cach XOR
        # thu cong - kiem tra dung logic ham is_duplicate_hash, khong phu
        # thuoc vao viec dung anh nao tao ra hash do.
        hash_a = 0b0000_0000
        hash_b = 0b1111_1111  # lech toi da 8 bit
        assert not is_duplicate_hash(hash_a, hash_b, threshold=4)

    def test_trong_nguong_hamming_duoc_coi_la_trung(self):
        hash_a = 0b0000_0000
        hash_b = 0b0000_0011  # lech 2 bit
        assert is_duplicate_hash(hash_a, hash_b, threshold=4)


class TestApplyStaticRegionDedup:
    def test_khong_khai_bao_static_region_nao_thi_giu_nguyen_ket_qua_engine(self):
        image = _solid_image((0, 0, 0))
        raw_regions = [RawTextRegion(text="abc", bbox=(0, 0, 50, 50), confidence=0.9)]
        cache = StaticRegionDedupCache()

        result = apply_static_region_dedup(image, raw_regions, static_regions=(), cache=cache)

        assert result == raw_regions

    def test_2_frame_giong_het_nhau_o_vung_static_frame_2_duoc_tai_su_dung_tu_cache(self):
        # Frame 1: vung logo (goc tren-trai) mau co dinh, engine doc duoc "DAI ABC".
        frame1 = _solid_image((0, 0, 0))
        frame1[0:60, 0:60] = (200, 200, 200)  # vung logo sang mau, phan con lai den
        region_logo = RawTextRegion(text="DAI ABC", bbox=(0, 0, 60, 60), confidence=0.9)
        cache = StaticRegionDedupCache()

        result1 = apply_static_region_dedup(
            frame1, [region_logo], static_regions=(LOGO_REGION,), cache=cache
        )
        assert result1 == [region_logo]
        assert len(cache) == 1

        # Frame 2: vung logo giong het frame 1 (cung mau, cung vi tri) nhung
        # engine (do nhieu nen/anh huong nen) doc lech 1 chut thanh "DAI ABC ".
        frame2 = frame1.copy()
        region_logo_noisy = RawTextRegion(text="DAI ABC ", bbox=(0, 0, 60, 60), confidence=0.7)

        result2 = apply_static_region_dedup(
            frame2, [region_logo_noisy], static_regions=(LOGO_REGION,), cache=cache
        )

        # Ket qua phai la ban da cache tu frame 1 (on dinh), khong phai ban
        # engine vua doc lech o frame 2.
        assert result2 == [region_logo]
        # Khong ghi them 1 entry moi vao cache vi da coi la trung.
        assert len(cache) == 1

    def test_vung_ticker_thay_doi_noi_dung_nhung_cung_layout_khong_bi_dedup(self):
        # Ticker o day CO CHU Y khong duoc khai bao trong static_regions -
        # dung logic ma OCRExtractor se dung thuc te (chi khai bao logo tinh
        # trong config, khong khai bao vung ticker).
        image_frame1 = _solid_image((10, 10, 10))
        image_frame2 = _solid_image((10, 10, 10))

        ticker_frame1 = RawTextRegion(text="Tin tuc so 1", bbox=(0, 150, 200, 200), confidence=0.9)
        ticker_frame2 = RawTextRegion(text="Tin tuc so 2 hoan toan khac", bbox=(0, 150, 200, 200), confidence=0.9)

        cache = StaticRegionDedupCache()
        result1 = apply_static_region_dedup(
            image_frame1, [ticker_frame1], static_regions=(LOGO_REGION,), cache=cache
        )
        result2 = apply_static_region_dedup(
            image_frame2, [ticker_frame2], static_regions=(LOGO_REGION,), cache=cache
        )

        # Ca 2 frame deu giu nguyen text moi nhat cua rieng minh - khong bi
        # ghi de boi cache (vi ticker khong nam trong static_regions).
        assert result1 == [ticker_frame1]
        assert result2 == [ticker_frame2]
        assert result1[0].text != result2[0].text

    def test_vung_khong_nam_trong_bat_ky_static_region_nao_luon_giu_nguyen(self):
        image = _solid_image((5, 5, 5))
        outside_region = RawTextRegion(text="noi dung giua man hinh", bbox=(300, 300, 350, 320), confidence=0.9)
        cache = StaticRegionDedupCache()

        result = apply_static_region_dedup(
            image, [outside_region], static_regions=(LOGO_REGION,), cache=cache
        )

        assert result == [outside_region]
        assert len(cache) == 0


class TestStaticRegionDedupCache:
    def test_lookup_tra_ve_none_khi_chua_co_entry_nao(self):
        cache = StaticRegionDedupCache()
        assert cache.lookup("logo", perceptual_hash(_solid_image((1, 2, 3)))) is None

    def test_update_roi_lookup_tim_thay_dung_entry(self):
        cache = StaticRegionDedupCache()
        image = _solid_image((9, 9, 9))
        image_hash = perceptual_hash(image)
        regions = [RawTextRegion(text="abc", bbox=(0, 0, 10, 10))]

        cache.update("logo", image_hash, regions)

        assert cache.lookup("logo", image_hash) == regions
