"""Test ``CaptionGenerator.caption_frame`` — VLM thật được thay bằng fake client.

Chỉ phần orchestration (đọc ảnh -> gọi VLM -> parse output -> đóng gói
CaptionResult) chạy thật; không cần tải model Qwen2-VL / GPU (cùng cách tiếp
cận với ``tests/shot_extractor/test_shot_extractor_wiring.py``: mock phần
nặng, giữ nguyên phần orchestration + logic thuần).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from BackEnd.app.caption.caption_module import CaptionGenerator
from BackEnd.app.contracts.pipeline import FrameMetadata


class _FakeVLMClient:
    """Thay thế ``VLMClient`` thật: trả về output cố định, ghi lại lời gọi để assert."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.use_4bit = False
        self.calls: list[dict] = []

    def generate(self, prompt_text: str, images=None) -> str:
        self.calls.append({"prompt_text": prompt_text, "images": images})
        return self.response


def _make_temp_frame(directory: Path, frame_id: str = "L21_V001_E001") -> FrameMetadata:
    image_path = directory / f"{frame_id}.jpg"
    Image.new("RGB", (16, 16), color="red").save(image_path)
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V001",
        shot_id="L21_V001_S000",
        timestamp_ms=1000,
        fps=25.0,
        frame_idx=25,
        frame_role="keyframe",
        source="official",
        frame_path=image_path,
    )


class CaptionFrameTests(unittest.TestCase):
    def test_returns_caption_result_with_frame_id_set(self) -> None:
        fake_response = (
            "Một người đàn ông mặc áo đỏ đang phát biểu.\n"
            "```json\n"
            '{"scene": "họp báo", "objects": ["person"], "actions": ["speaking"]}\n'
            "```"
        )
        fake_client = _FakeVLMClient(response=fake_response)
        generator = CaptionGenerator(vlm_client=fake_client)

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame = _make_temp_frame(Path(tmp_dir))
            results = generator.caption_frame(frame)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.frame_id, frame.frame_id)
        self.assertIsNone(result.clip_id)
        self.assertIsNone(result.shot_id)
        self.assertEqual(result.caption_text, "Một người đàn ông mặc áo đỏ đang phát biểu.")
        self.assertEqual(result.structured_data["scene"], "họp báo")
        self.assertEqual(result.model_name, "Qwen2-VL-7B-Instruct")

        # 1 ảnh duy nhất phải được truyền vào VLM cho frame caption.
        self.assertEqual(len(fake_client.calls), 1)
        self.assertEqual(len(fake_client.calls[0]["images"]), 1)

    def test_returns_empty_list_when_frame_path_missing(self) -> None:
        fake_client = _FakeVLMClient(response="không nên tới đây")
        generator = CaptionGenerator(vlm_client=fake_client)
        frame = FrameMetadata(
            frame_id="L21_V001_E002",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=1000,
            fps=25.0,
            frame_idx=25,
            frame_role="keyframe",
            source="official",
            frame_path=None,
        )

        results = generator.caption_frame(frame)

        self.assertEqual(results, [])
        self.assertEqual(fake_client.calls, [])

    def test_returns_empty_list_when_image_file_does_not_exist(self) -> None:
        fake_client = _FakeVLMClient(response="không nên tới đây")
        generator = CaptionGenerator(vlm_client=fake_client)
        frame = FrameMetadata(
            frame_id="L21_V001_E003",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=1000,
            fps=25.0,
            frame_idx=25,
            frame_role="keyframe",
            source="official",
            frame_path=Path("this/path/does/not/exist.jpg"),
        )

        results = generator.caption_frame(frame)

        self.assertEqual(results, [])
        self.assertEqual(fake_client.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
