"""Test ``OCRExtractor`` (module_ocr.md mục 12.2) — dùng ``FakeOCREngine``, không cần model thật."""

from __future__ import annotations

from pathlib import Path

import pytest

from BackEnd.app.ocr.config import OCRConfig
from BackEnd.app.ocr.engines.base import RawTextRegion
from BackEnd.app.ocr.ocr_extractor import OCRExtractor


def _region(text: str, bbox=(10, 10, 60, 40), confidence: float = 0.9) -> RawTextRegion:
    return RawTextRegion(text=text, bbox=bbox, confidence=confidence)


class TestExtractFrameCoChu:
    def test_frame_co_chu_tra_ve_dung_so_luong_va_n_khong_trung(
        self, fake_engine_factory, make_frame_metadata
    ):
        # clear_text.jpg (fixture mac dinh cua make_frame_metadata) kich
        # thuoc 400x200 (rong x cao) - cac bbox phai nam trong 200px chieu
        # cao va cach nhau > line_y_tolerance de khong bi gop thanh 1 hang.
        regions = [
            _region("Dong 1", bbox=(0, 0, 50, 20)),
            _region("Dong 2", bbox=(0, 80, 50, 100)),
            _region("Dong 3", bbox=(0, 160, 50, 180)),
        ]
        engine = fake_engine_factory(regions=regions)
        extractor = OCRExtractor(engine=engine)
        frame = make_frame_metadata()

        results = extractor.extract(frame)

        assert len(results) == len(regions)
        ns = [r.n for r in results]
        assert ns == list(range(len(regions)))
        assert len(set(ns)) == len(ns)

    def test_moi_ocrresult_tra_ve_dung_frame_id_dau_vao(self, fake_engine_factory, make_frame_metadata):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frame = make_frame_metadata(frame_id="F999999")

        results = extractor.extract(frame)

        assert all(r.frame_id == "F999999" for r in results)

    def test_bbox_tra_ve_normalized_trong_khoang_0_1(self, fake_engine_factory, make_frame_metadata):
        engine = fake_engine_factory(regions=[_region("abc", bbox=(10, 10, 60, 40))])
        extractor = OCRExtractor(engine=engine)

        results = extractor.extract(make_frame_metadata())

        assert len(results) == 1
        result = results[0]
        assert 0.0 <= result.x_min < result.x_max <= 1.0
        assert 0.0 <= result.y_min < result.y_max <= 1.0


class TestExtractFrameKhongCoChu:
    def test_engine_tra_ve_rong_thi_extract_tra_ve_rong_khong_raise(
        self, fake_engine_factory, make_frame_metadata
    ):
        engine = fake_engine_factory(regions=[])
        extractor = OCRExtractor(engine=engine)

        results = extractor.extract(make_frame_metadata())

        assert results == []


class TestExtractValidation:
    def test_frame_path_none_raise_loi_ro_rang(self, fake_engine_factory, make_frame_metadata):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frame = make_frame_metadata(frame_path=None)

        with pytest.raises(ValueError):
            extractor.extract(frame)

    def test_frame_path_khong_ton_tai_raise_loi_ro_rang(
        self, fake_engine_factory, make_frame_metadata, tmp_path: Path
    ):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frame = make_frame_metadata(frame_path=tmp_path / "khong_ton_tai.jpg")

        with pytest.raises(FileNotFoundError):
            extractor.extract(frame)

    def test_frame_role_khac_keyframe_bi_tu_choi(self, fake_engine_factory, make_frame_metadata):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frame = make_frame_metadata(frame_role="tracking_sample")

        with pytest.raises(ValueError):
            extractor.extract(frame)

        # Khong duoc goi den engine khi da tu choi tu buoc validate frame_role.
        assert engine.call_count == 0


class TestExtractBatch:
    def test_extract_batch_tra_ve_du_key_cho_moi_frame(self, fake_engine_factory, make_frame_metadata):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frames = [make_frame_metadata(frame_id=f"F{i:06d}") for i in range(5)]

        results = extractor.extract_batch(frames)

        assert set(results.keys()) == {frame.frame_id for frame in frames}
        for frame_id, frame_results in results.items():
            assert all(r.frame_id == frame_id for r in frame_results)

    def test_1_frame_loi_giua_batch_khong_lam_crash_toan_batch(
        self, fake_engine_factory, make_frame_metadata, tmp_path: Path
    ):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)

        good_frame_1 = make_frame_metadata(frame_id="F000001")
        broken_frame = make_frame_metadata(frame_id="F000002", frame_path=tmp_path / "missing.jpg")
        good_frame_2 = make_frame_metadata(frame_id="F000003")

        results = extractor.extract_batch([good_frame_1, broken_frame, good_frame_2])

        assert len(results) == 3
        assert results["F000002"] == []
        assert len(results["F000001"]) == 1
        assert len(results["F000003"]) == 1

    def test_thu_tu_frame_trong_danh_sach_khong_anh_huong_ket_qua_tung_frame(
        self, fake_engine_factory, make_frame_metadata
    ):
        engine = fake_engine_factory(regions=[_region("abc")])
        extractor = OCRExtractor(engine=engine)
        frame_a = make_frame_metadata(frame_id="A")
        frame_b = make_frame_metadata(frame_id="B")

        forward = extractor.extract_batch([frame_a, frame_b])
        backward = extractor.extract_batch([frame_b, frame_a])

        assert [r.text for r in forward["A"]] == [r.text for r in backward["A"]]
        assert [r.text for r in forward["B"]] == [r.text for r in backward["B"]]


class TestExtractorConfig:
    def test_dung_config_tuy_chinh_thay_vi_mac_dinh(self, fake_engine_factory, make_frame_metadata):
        # confidence_threshold rat cao -> vung confidence 0.9 (mac dinh trong
        # _region) van qua nguong, nhung dat 0.95 se bi loc bo - kiem tra
        # OCRConfig truyen vao thuc su duoc ap dung.
        engine = fake_engine_factory(regions=[_region("abc", confidence=0.9)])
        custom_config = OCRConfig(confidence_threshold=0.95)
        extractor = OCRExtractor(engine=engine, config=custom_config)

        results = extractor.extract(make_frame_metadata())

        assert results == []
