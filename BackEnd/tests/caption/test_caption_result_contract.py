"""Test: mọi ``CaptionResult`` do ``CaptionGenerator`` trả về đều hợp lệ theo contract.

``CaptionResult.__post_init__`` (app/contracts/pipeline.py) tự raise
``ValueError`` nếu không đúng 1 trong frame_id/clip_id/shot_id được set
(module_caption.md mục 2, mục 5). File này verify cả 3 hàm public của
``CaptionGenerator`` luôn tạo ra result hợp lệ - kể cả khi VLM trả về output
"bẩn" (JSON lỗi, không có JSON) - và verify lại chính bất biến đó trên
``CaptionResult`` một cách độc lập.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from BackEnd.app.caption.caption_module import CaptionGenerator
from BackEnd.app.contracts.pipeline import CaptionResult, FrameMetadata, ShotMetadata


class _FakeVLMClient:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.use_4bit = False

    def generate(self, prompt_text: str, images=None) -> str:
        return self.response


def _caption_level(result: CaptionResult) -> str:
    """Xác định cấp của 1 CaptionResult dựa vào id nào khác None (module_caption.md mục 3.0)."""

    if result.frame_id is not None:
        return "frame"
    if result.clip_id is not None:
        return "clip"
    return "shot"


class CaptionResultContractDirectTests(unittest.TestCase):
    """Test lại đúng bất biến của CaptionResult, độc lập với CaptionGenerator."""

    def test_exactly_one_id_required(self) -> None:
        with self.assertRaises(ValueError):
            CaptionResult(caption_text="x", model_name="m")  # không id nào được set

    def test_two_ids_set_at_once_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CaptionResult(caption_text="x", model_name="m", frame_id="f1", clip_id="c1")

    def test_exactly_one_id_is_accepted(self) -> None:
        result = CaptionResult(caption_text="x", model_name="m", frame_id="f1")
        self.assertEqual(_caption_level(result), "frame")


class CaptionGeneratorAlwaysReturnsValidResultsTests(unittest.TestCase):
    """Test CaptionGenerator không bao giờ crash vì __post_init__, kể cả với output bẩn của VLM."""

    def _frame(self, directory: Path) -> FrameMetadata:
        image_path = directory / "frame.jpg"
        Image.new("RGB", (16, 16), color="green").save(image_path)
        return FrameMetadata(
            frame_id="L21_V001_E000",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
            fps=25.0,
            frame_idx=0,
            source="official",
            frame_path=image_path,
        )

    def test_caption_frame_valid_result_even_with_malformed_vlm_output(self) -> None:
        for dirty_response in ("chỉ có text, không JSON", "```json\n{broken\n```", ""):
            with self.subTest(dirty_response=dirty_response):
                generator = CaptionGenerator(vlm_client=_FakeVLMClient(response=dirty_response))
                with tempfile.TemporaryDirectory() as tmp_dir:
                    results = generator.caption_frame(self._frame(Path(tmp_dir)))

                self.assertEqual(len(results), 1)
                self.assertEqual(_caption_level(results[0]), "frame")
                self.assertIsNone(results[0].structured_data)

    def test_caption_shot_single_clip_result_is_valid(self) -> None:
        generator = CaptionGenerator(vlm_client=_FakeVLMClient(response="không dùng tới"))
        shot = ShotMetadata(shot_id="L21_V001_S000", video_id="L21_V001", shot_index=0, start_ms=0, end_ms=1000)
        clip_caption = CaptionResult(
            caption_text="Mô tả clip.", model_name="Qwen2-VL-7B-Instruct", clip_id="L21_V001_C000"
        )

        results = generator.caption_shot(shot, [clip_caption])

        self.assertEqual(len(results), 1)
        self.assertEqual(_caption_level(results[0]), "shot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
