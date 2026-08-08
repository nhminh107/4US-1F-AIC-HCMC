"""Test ``parsing.parse_caption_output`` / ``validate_structured_data``.

Logic thuần, không cần VLM/GPU — mô phỏng các kiểu output thô mà VLM có thể
trả về (đúng định dạng, thiếu code fence, JSON lỗi, không có JSON).
"""

from __future__ import annotations

import unittest

from BackEnd.app.caption.parsing import parse_caption_output, validate_structured_data


class ParseCaptionOutputTests(unittest.TestCase):
    def test_parses_well_formed_code_fence_json(self) -> None:
        raw = (
            "Một người đàn ông mặc áo đỏ đang phát biểu trước đám đông.\n"
            "```json\n"
            '{"scene": "họp báo", "objects": ["person", "microphone"], "actions": ["speaking"]}\n'
            "```"
        )

        caption_text, structured_data = parse_caption_output(raw)

        self.assertEqual(
            caption_text, "Một người đàn ông mặc áo đỏ đang phát biểu trước đám đông."
        )
        self.assertEqual(structured_data["scene"], "họp báo")
        self.assertEqual(structured_data["objects"], ["person", "microphone"])

    def test_falls_back_to_bare_json_without_code_fence(self) -> None:
        raw = 'Một cảnh biển đẹp.\n{"scene": "biển", "objects": [], "actions": []}'

        caption_text, structured_data = parse_caption_output(raw)

        self.assertEqual(caption_text, "Một cảnh biển đẹp.")
        self.assertEqual(structured_data, {"scene": "biển", "objects": [], "actions": []})

    def test_invalid_json_keeps_caption_text_and_returns_none_structured_data(self) -> None:
        # Cặp {} cân bằng (để regex bắt được khối) nhưng nội dung bên trong sai
        # cú pháp JSON (thiếu dấu ngoặc kép quanh giá trị) -> json.loads() lỗi.
        raw = "Mô tả hợp lệ.\n```json\n{scene: not_quoted_value}\n```"

        caption_text, structured_data = parse_caption_output(raw)

        self.assertEqual(caption_text, "Mô tả hợp lệ.")
        self.assertIsNone(structured_data)

    def test_no_json_block_returns_entire_text_as_caption(self) -> None:
        raw = "Chỉ có mô tả tự do, không có JSON gì cả."

        caption_text, structured_data = parse_caption_output(raw)

        self.assertEqual(caption_text, raw)
        self.assertIsNone(structured_data)

    def test_json_only_output_falls_back_to_raw_text_as_caption(self) -> None:
        raw = '```json\n{"scene": "x", "objects": [], "actions": []}\n```'

        caption_text, structured_data = parse_caption_output(raw)

        # Không được trả caption_text rỗng dù model chỉ sinh JSON (cột NOT NULL).
        self.assertTrue(caption_text)
        self.assertEqual(structured_data["scene"], "x")

    def test_strips_surrounding_whitespace(self) -> None:
        raw = "\n\n  Mô tả có khoảng trắng thừa.  \n\n"

        caption_text, _ = parse_caption_output(raw)

        self.assertEqual(caption_text, "Mô tả có khoảng trắng thừa.")


class ValidateStructuredDataTests(unittest.TestCase):
    def test_accepts_dict_with_all_required_keys(self) -> None:
        self.assertTrue(
            validate_structured_data({"scene": "s", "objects": [], "actions": [], "attributes": {}})
        )

    def test_rejects_missing_required_key(self) -> None:
        self.assertFalse(validate_structured_data({"scene": "s", "objects": []}))

    def test_rejects_non_dict(self) -> None:
        self.assertFalse(validate_structured_data(None))
        self.assertFalse(validate_structured_data("not a dict"))

    def test_allows_extra_fields_beyond_required(self) -> None:
        self.assertTrue(
            validate_structured_data(
                {"scene": "s", "objects": [], "actions": [], "extra_field": "ok"}
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
