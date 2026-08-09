"""Test riêng ``MonkeyOCREngine`` (module_ocr.md mục 12.3).

Sau khi xác nhận kiến trúc thật của MonkeyOCRv2-Parsing (đọc trực tiếp
``github.com/Yuliang-Liu/MonkeyOCRv2``), engine là 1 **HTTP client** gọi tới
vLLM server, không phải model nạp local — nên phần lớn test ở đây mock
``requests.Session.post`` (không cần server/GPU thật). Test cần 1 vLLM server
thật đang chạy (``python serve.py``, xem module_ocr.md mục 11/`ocr_pipeline.md`)
được đánh dấu ``@pytest.mark.gpu`` và skip mặc định.
"""

from __future__ import annotations

import json

import pytest
import requests

from BackEnd.app.ocr import config
from BackEnd.app.ocr.engines.base import RawTextRegion
from BackEnd.app.ocr.engines.monkeyocr_engine import (
    MonkeyOCREngine,
    _extract_json_items,
    _image_to_png_data_uri,
    _map_bbox_to_pixels,
    _normalize_item,
    _parse_end2end_output,
)
from BackEnd.tests.ocr.conftest import EXPECTED_DIR, IMAGES_DIR


class TestExtractJsonItems:
    def test_json_array_thuan_tuy_duoc_parse_dung(self):
        raw = '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "abc"}]'
        items = _extract_json_items(raw)
        assert items == [{"bbox": [0, 0, 100, 100], "label": "Text", "content": "abc"}]

    def test_json_boc_trong_markdown_code_fence_van_parse_duoc(self):
        raw = '```json\n[{"bbox": [0, 0, 10, 10], "content": "x"}]\n```'
        items = _extract_json_items(raw)
        assert len(items) == 1

    def test_co_text_thua_truoc_sau_json_van_phuc_hoi_duoc(self):
        raw = 'Day la ket qua:\n[{"bbox": [1, 2, 3, 4], "content": "y"}]\nHet.'
        items = _extract_json_items(raw)
        assert len(items) == 1

    def test_json_hong_khong_the_phuc_hoi_tra_ve_rong(self):
        assert _extract_json_items("khong phai json chut nao") == []

    def test_chuoi_rong_tra_ve_rong(self):
        assert _extract_json_items("") == []

    def test_json_khong_phai_list_tra_ve_rong(self):
        assert _extract_json_items('{"bbox": [0, 0, 1, 1]}') == []


class TestNormalizeItem:
    def test_item_hop_le_duoc_giu_lai(self):
        result = _normalize_item({"bbox": [1, 2, 3, 4], "content": "abc", "label": "Text"})
        assert result == {"bbox": [1.0, 2.0, 3.0, 4.0], "content": "abc"}

    def test_thieu_bbox_tra_ve_none(self):
        assert _normalize_item({"content": "abc"}) is None

    def test_bbox_sai_do_dai_tra_ve_none(self):
        assert _normalize_item({"bbox": [1, 2, 3], "content": "abc"}) is None

    def test_bbox_khong_phai_so_tra_ve_none(self):
        assert _normalize_item({"bbox": ["a", "b", "c", "d"], "content": "abc"}) is None

    def test_khong_phai_dict_tra_ve_none(self):
        assert _normalize_item(["bbox", [1, 2, 3, 4]]) is None

    def test_thieu_content_mac_dinh_chuoi_rong(self):
        result = _normalize_item({"bbox": [0, 0, 1, 1]})
        assert result["content"] == ""


class TestMapBboxToPixels:
    def test_quy_doi_dung_cong_thuc_chia_1000_nhan_kich_thuoc_anh(self):
        pixel_bbox = _map_bbox_to_pixels([0, 0, 500, 1000], image_width=200, image_height=100)
        assert pixel_bbox == (0.0, 0.0, 100.0, 100.0)

    def test_bbox_full_khung_1000_tra_ve_dung_kich_thuoc_anh_that(self):
        pixel_bbox = _map_bbox_to_pixels([0, 0, 1000, 1000], image_width=400, image_height=200)
        assert pixel_bbox == (0.0, 0.0, 400.0, 200.0)


class TestParseEnd2EndOutput:
    def test_parse_dung_thanh_rawtextregion_toa_do_pixel(self):
        raw_text = json.dumps(
            [{"bbox": [0, 0, 500, 500], "label": "Text", "content": "Xin chao"}]
        )
        regions = _parse_end2end_output(raw_text, image_width=200, image_height=100)

        assert regions == [RawTextRegion(text="Xin chao", bbox=(0.0, 0.0, 100.0, 50.0))]

    def test_block_khong_co_content_bi_bo_qua(self):
        raw_text = json.dumps([{"bbox": [0, 0, 10, 10], "label": "Picture", "content": ""}])
        assert _parse_end2end_output(raw_text, image_width=100, image_height=100) == []

    def test_output_khong_parse_duoc_tra_ve_rong_khong_raise(self):
        assert _parse_end2end_output("loi server tra ve linh tinh", image_width=100, image_height=100) == []

    def test_khong_co_confidence_va_language_tu_server(self):
        raw_text = json.dumps([{"bbox": [0, 0, 100, 100], "content": "abc"}])
        regions = _parse_end2end_output(raw_text, image_width=100, image_height=100)
        assert regions[0].confidence is None
        assert regions[0].language is None


class TestImageToPngDataUri:
    def test_tra_ve_dung_dinh_dang_data_uri_png(self):
        from PIL import Image

        image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        data_uri = _image_to_png_data_uri(image)
        assert data_uri.startswith("data:image/png;base64,")


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict:
        return self._payload


class TestMonkeyOCREngineRunWithMockedServer:
    """Mock ``requests.Session.post`` — không cần vLLM server/GPU thật."""

    def test_anh_none_tra_ve_rong_khong_goi_http(self, monkeypatch):
        engine = MonkeyOCREngine(server_url="http://fake-server:8888")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("Không được gọi HTTP khi ảnh rỗng")

        monkeypatch.setattr(engine._session, "post", _fail_if_called)
        assert engine.run(None) == []

    def test_run_goi_dung_endpoint_va_parse_dung_response(self, monkeypatch):
        import numpy as np

        captured_requests = []

        def _fake_post(url, json=None, timeout=None):
            captured_requests.append({"url": url, "json": json, "timeout": timeout})
            content = json_module.dumps(
                [{"bbox": [0, 0, 1000, 1000], "label": "Text", "content": "Xin chao Viet Nam"}]
            )
            return _FakeResponse({"choices": [{"message": {"content": content}}]})

        import json as json_module

        engine = MonkeyOCREngine(server_url="http://fake-server:8888")
        monkeypatch.setattr(engine._session, "post", _fake_post)

        image = np.zeros((50, 100, 3), dtype="uint8")
        regions = engine.run(image)

        assert len(captured_requests) == 1
        assert captured_requests[0]["url"] == "http://fake-server:8888/v1/chat/completions"
        sent_payload = captured_requests[0]["json"]
        assert sent_payload["messages"][0]["content"][1]["text"] == config.END2END_PROMPT

        assert len(regions) == 1
        assert regions[0].text == "Xin chao Viet Nam"
        assert regions[0].bbox == (0.0, 0.0, 100.0, 50.0)

    def test_server_khong_ket_noi_duoc_raise_loi_ro_rang(self, monkeypatch):
        import numpy as np

        def _fake_post(*args, **kwargs):
            raise requests.ConnectionError("connection refused")

        engine = MonkeyOCREngine(server_url="http://fake-server:8888")
        monkeypatch.setattr(engine._session, "post", _fake_post)

        with pytest.raises(RuntimeError):
            engine.run(np.zeros((10, 10, 3), dtype="uint8"))


# ---------------------------------------------------------------------------
# Test cần 1 vLLM server MonkeyOCRv2 THẬT đang chạy (module_ocr.md mục 12.3,
# 12.9; xem hướng dẫn khởi động ở `Markdown_Doc/ocr_pipeline.md`). Bị skip
# mặc định (`pytest tests/ocr -m "not gpu"`), chỉ chạy với
# `pytest tests/ocr -m gpu` sau khi đã `python serve.py` trên máy có GPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestMonkeyOCREngineWithRealServer:
    def test_ket_noi_toi_vllm_server_thanh_cong(self):
        import cv2

        engine = MonkeyOCREngine()
        image = cv2.imread(str(IMAGES_DIR / "clear_text.jpg"))
        # Không raise RuntimeError (lỗi kết nối) tức là server đang chạy đúng.
        engine.run(image)

    def test_anh_co_chu_tieng_viet_co_dau_giu_dung_dau(self):
        import cv2

        expected = json.loads((EXPECTED_DIR / "clear_text.json").read_text(encoding="utf-8"))
        image = cv2.imread(str(IMAGES_DIR / expected["image"]))

        engine = MonkeyOCREngine()
        regions = engine.run(image)

        texts = [region.text for region in regions]
        assert any(
            expected_region["text"] in text
            for expected_region in expected["expected_regions"]
            for text in texts
        )

    def test_anh_khong_co_chu_tra_ve_danh_sach_rong(self, no_text_image_path):
        import cv2

        image = cv2.imread(str(no_text_image_path))
        engine = MonkeyOCREngine()
        assert engine.run(image) == []

    def test_anh_co_ticker_van_detect_va_recognize_duoc(self, ticker_banner_image_path):
        import cv2

        image = cv2.imread(str(ticker_banner_image_path))
        engine = MonkeyOCREngine()
        regions = engine.run(image)
        assert len(regions) > 0
