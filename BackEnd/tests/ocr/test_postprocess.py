"""Test ``app/ocr/postprocess.py`` (module_ocr.md mục 12.7)."""

from __future__ import annotations

from BackEnd.app.ocr.config import CONFIDENCE_THRESHOLD
from BackEnd.app.ocr.engines.base import RawTextRegion
from BackEnd.app.ocr.postprocess import (
    filter_by_confidence,
    merge_same_line_regions,
    normalize_text,
    postprocess,
    sort_regions_reading_order,
)


class TestNormalizeText:
    def test_xoa_khoang_trang_thua_dau_cuoi_va_giua(self):
        assert normalize_text("   Xin   chao   ") == "Xin chao"

    def test_xoa_ky_tu_dieu_khien_la(self):
        assert normalize_text("Xin\x00chao\x0bVN") == "XinchaoVN"

    def test_chuoi_rong_van_tra_ve_chuoi_rong(self):
        assert normalize_text("   ") == ""


class TestSortRegionsReadingOrder:
    def test_sap_theo_hang_tren_truoc_duoi_sau(self):
        top = RawTextRegion(text="tren", bbox=(0, 0, 50, 20))
        bottom = RawTextRegion(text="duoi", bbox=(0, 100, 50, 120))

        ordered = sort_regions_reading_order([bottom, top])

        assert [r.text for r in ordered] == ["tren", "duoi"]

    def test_cung_hang_sap_trai_truoc_phai_sau(self):
        right = RawTextRegion(text="phai", bbox=(100, 0, 150, 20))
        left = RawTextRegion(text="trai", bbox=(0, 0, 50, 20))

        ordered = sort_regions_reading_order([right, left])

        assert [r.text for r in ordered] == ["trai", "phai"]


class TestMergeSameLineRegions:
    def test_gop_cac_manh_cung_hang_thanh_1_vung_giu_dung_thu_tu_doc(self):
        part1 = RawTextRegion(text="Xin", bbox=(0, 0, 30, 20), confidence=0.9)
        part2 = RawTextRegion(text="chao", bbox=(35, 2, 70, 20), confidence=0.8)

        merged = merge_same_line_regions([part2, part1])

        assert len(merged) == 1
        assert merged[0].text == "Xin chao"
        # Bbox gop phai la hop (union) cua 2 vung con.
        assert merged[0].bbox == (0, 0, 70, 20)
        # Confidence cua vung gop = gia tri thap nhat trong nhom.
        assert merged[0].confidence == 0.8

    def test_khac_hang_khong_bi_gop(self):
        top = RawTextRegion(text="tren", bbox=(0, 0, 50, 20))
        bottom = RawTextRegion(text="duoi", bbox=(0, 100, 50, 120))

        merged = merge_same_line_regions([top, bottom])

        assert [r.text for r in merged] == ["tren", "duoi"]


class TestFilterByConfidence:
    def test_loai_vung_co_confidence_thap_hon_nguong(self):
        low = RawTextRegion(text="mo", bbox=(0, 0, 10, 10), confidence=0.1)
        high = RawTextRegion(text="ro", bbox=(0, 0, 10, 10), confidence=0.9)

        filtered = filter_by_confidence([low, high], threshold=CONFIDENCE_THRESHOLD)

        assert filtered == [high]

    def test_khong_co_confidence_thi_khong_bi_loai(self):
        unknown = RawTextRegion(text="unknown", bbox=(0, 0, 10, 10), confidence=None)
        filtered = filter_by_confidence([unknown], threshold=CONFIDENCE_THRESHOLD)
        assert filtered == [unknown]


class TestPostprocessPipeline:
    def test_gop_chuan_hoa_va_loc_theo_dung_thu_tu(self):
        regions = [
            RawTextRegion(text="  Xin  ", bbox=(0, 0, 30, 20), confidence=0.9),
            RawTextRegion(text="chao", bbox=(35, 2, 70, 20), confidence=0.9),
            RawTextRegion(text="rac", bbox=(0, 100, 30, 120), confidence=0.05),
        ]

        result = postprocess(regions, confidence_threshold=CONFIDENCE_THRESHOLD)

        assert len(result) == 1
        assert result[0].text == "Xin chao"

    def test_khong_co_vung_nao_tra_ve_danh_sach_rong(self):
        assert postprocess([]) == []
