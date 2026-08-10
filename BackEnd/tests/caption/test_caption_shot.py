"""Test ``CaptionGenerator.caption_shot`` — 3 nhánh: 0 / 1 / >=2 clip caption.

Không cần VLM thật cho nhánh 0 và 1 clip (không gọi VLM); nhánh >=2 dùng fake
client để verify lời gọi text-only (không kèm ảnh), đúng nguyên tắc "không
chạy lại VLM trên ảnh ở cấp shot" (module_caption.md mục 3.3).
"""

from __future__ import annotations

import unittest

from BackEnd.app.caption.caption_module import CaptionGenerator
from BackEnd.app.contracts.pipeline import CaptionResult, ShotMetadata


class _FakeVLMClient:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.use_4bit = False
        self.calls: list[dict] = []

    def generate(self, prompt_text: str, images=None) -> str:
        self.calls.append({"prompt_text": prompt_text, "images": images})
        return self.response


def _shot() -> ShotMetadata:
    return ShotMetadata(shot_id="L21_V001_S000", video_id="L21_V001", shot_index=0, start_ms=0, end_ms=5000)


def _clip_caption(clip_id: str, text: str) -> CaptionResult:
    return CaptionResult(
        caption_text=text,
        model_name="Qwen2-VL-7B-Instruct",
        clip_id=clip_id,
        structured_data={"scene": "s", "objects": ["person"], "actions": ["walking"]},
        model_version="fp16",
        prompt_version="v1",
    )


class CaptionShotTests(unittest.TestCase):
    def test_returns_empty_list_when_no_clip_captions(self) -> None:
        fake_client = _FakeVLMClient(response="không nên tới đây")
        generator = CaptionGenerator(vlm_client=fake_client)

        results = generator.caption_shot(_shot(), [])

        self.assertEqual(results, [])
        self.assertEqual(fake_client.calls, [])

    def test_single_clip_caption_is_reused_without_calling_vlm(self) -> None:
        fake_client = _FakeVLMClient(response="không nên tới đây")
        generator = CaptionGenerator(vlm_client=fake_client)
        clip_caption = _clip_caption("L21_V001_C000", "Người đàn ông bước lên bục.")

        results = generator.caption_shot(_shot(), [clip_caption])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.shot_id, "L21_V001_S000")
        self.assertIsNone(result.clip_id)
        self.assertIsNone(result.frame_id)
        # Nội dung phải giống hệt clip caption gốc - không gọi lại VLM.
        self.assertEqual(result.caption_text, clip_caption.caption_text)
        self.assertEqual(result.structured_data, clip_caption.structured_data)
        self.assertEqual(result.model_version, clip_caption.model_version)
        self.assertEqual(fake_client.calls, [])

    def test_multiple_clip_captions_trigger_text_only_summarization(self) -> None:
        fake_response = (
            "Cảnh quay buổi họp báo: người đàn ông bước lên bục và phát biểu.\n"
            "```json\n"
            '{"scene": "họp báo", "objects": ["person"], "actions": ["walking", "speaking"]}\n'
            "```"
        )
        fake_client = _FakeVLMClient(response=fake_response)
        generator = CaptionGenerator(vlm_client=fake_client)
        clip_captions = [
            _clip_caption("L21_V001_C000", "Người đàn ông bước lên bục."),
            _clip_caption("L21_V001_C001", "Người đàn ông bắt đầu phát biểu."),
        ]

        results = generator.caption_shot(_shot(), clip_captions)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.shot_id, "L21_V001_S000")
        self.assertEqual(result.caption_text, fake_response.splitlines()[0])

        self.assertEqual(len(fake_client.calls), 1)
        call = fake_client.calls[0]
        # Cấp shot KHÔNG được kèm ảnh - chỉ tổng hợp từ text các clip caption.
        self.assertIsNone(call["images"])
        self.assertIn("Người đàn ông bước lên bục.", call["prompt_text"])
        self.assertIn("Người đàn ông bắt đầu phát biểu.", call["prompt_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
