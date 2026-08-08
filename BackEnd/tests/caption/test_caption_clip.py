"""Test ``CaptionGenerator.caption_clip`` — VLM thật được thay bằng fake client."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from BackEnd.app.caption import config
from BackEnd.app.caption.caption_module import CaptionGenerator
from BackEnd.app.contracts.pipeline import ClipWindowMetadata, FrameMetadata


class _FakeVLMClient:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.use_4bit = False
        self.calls: list[dict] = []

    def generate(self, prompt_text: str, images=None) -> str:
        self.calls.append({"prompt_text": prompt_text, "images": images})
        return self.response


def _make_temp_frames(directory: Path, count: int, start_ms: int = 0, step_ms: int = 200) -> list[FrameMetadata]:
    frames = []
    for i in range(count):
        frame_id = f"L21_V001_E{i:03d}"
        timestamp_ms = start_ms + i * step_ms
        image_path = directory / f"{frame_id}.jpg"
        Image.new("RGB", (16, 16), color="blue").save(image_path)
        frames.append(
            FrameMetadata(
                frame_id=frame_id,
                video_id="L21_V001",
                shot_id="L21_V001_S000",
                timestamp_ms=timestamp_ms,
                fps=25.0,
                frame_idx=timestamp_ms // 40,
                frame_role="keyframe",
                source="official",
                frame_path=image_path,
            )
        )
    return frames


class CaptionClipTests(unittest.TestCase):
    def test_returns_caption_result_with_clip_id_set(self) -> None:
        fake_response = (
            "Người đàn ông bước lên bục và bắt đầu phát biểu.\n"
            "```json\n"
            '{"scene": "bục phát biểu", "objects": ["person"], "actions": ["walking", "speaking"]}\n'
            "```"
        )
        fake_client = _FakeVLMClient(response=fake_response)
        generator = CaptionGenerator(vlm_client=fake_client)
        clip = ClipWindowMetadata(clip_id="L21_V001_C000", shot_id="L21_V001_S000", start_ms=0, end_ms=1000)

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = _make_temp_frames(Path(tmp_dir), count=4)
            results = generator.caption_clip(clip, frames)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.clip_id, clip.clip_id)
        self.assertIsNone(result.frame_id)
        self.assertIsNone(result.shot_id)
        self.assertEqual(len(fake_client.calls), 1)
        # Toàn bộ 4 frame đại diện phải được feed cùng lúc vào 1 lần gọi VLM.
        self.assertEqual(len(fake_client.calls[0]["images"]), 4)
        # Prompt phải phản ánh đúng số ảnh thật sự truyền vào (thay <NUM_FRAMES>).
        self.assertIn("4 khung hình", fake_client.calls[0]["prompt_text"])

    def test_caps_number_of_frames_at_max_clip_sample_count(self) -> None:
        fake_client = _FakeVLMClient(response="mô tả bất kỳ")
        generator = CaptionGenerator(vlm_client=fake_client)
        clip = ClipWindowMetadata(clip_id="L21_V001_C001", shot_id="L21_V001_S000", start_ms=0, end_ms=10_000)

        too_many = config.MAX_CLIP_SAMPLE_COUNT + 5
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = _make_temp_frames(Path(tmp_dir), count=too_many, step_ms=100)
            results = generator.caption_clip(clip, frames)

        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(fake_client.calls[0]["images"]), config.MAX_CLIP_SAMPLE_COUNT)

    def test_returns_empty_list_when_no_sampled_frames(self) -> None:
        fake_client = _FakeVLMClient(response="không nên tới đây")
        generator = CaptionGenerator(vlm_client=fake_client)
        clip = ClipWindowMetadata(clip_id="L21_V001_C002", shot_id="L21_V001_S000", start_ms=0, end_ms=1000)

        results = generator.caption_clip(clip, [])

        self.assertEqual(results, [])
        self.assertEqual(fake_client.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
