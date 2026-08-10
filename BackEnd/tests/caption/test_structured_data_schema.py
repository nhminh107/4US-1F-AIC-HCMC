"""Test schema tối thiểu của ``structured_data`` (module_caption.md mục 4).

Khác với ``test_parsing.py`` (test hành vi parse output thô), file này tập
trung vào chính "hợp đồng" schema: field nào bắt buộc, ví dụ JSON trong tài
liệu đặc tả có hợp lệ không.
"""

from __future__ import annotations

import unittest

from BackEnd.app.caption.parsing import REQUIRED_STRUCTURED_KEYS, validate_structured_data


class RequiredStructuredKeysTests(unittest.TestCase):
    def test_required_keys_match_spec(self) -> None:
        # module_caption.md mục 4: "tối thiểu cần scene, objects, actions".
        self.assertEqual(set(REQUIRED_STRUCTURED_KEYS), {"scene", "objects", "actions"})

    def test_example_json_from_module_caption_doc_is_valid(self) -> None:
        # Ví dụ nguyên văn ở module_caption.md mục 4.
        example = {
            "scene": "cuộc họp báo ngoài trời",
            "objects": ["person", "microphone", "tree"],
            "actions": ["speaking", "standing"],
            "attributes": {"colors": ["red shirt"], "counts": {"person": 3}},
        }

        self.assertTrue(validate_structured_data(example))

    def test_missing_actions_is_invalid(self) -> None:
        example = {"scene": "s", "objects": ["person"]}
        self.assertFalse(validate_structured_data(example))

    def test_extra_field_beyond_spec_still_valid(self) -> None:
        # module_caption.md mục 4: "Field không cố định cứng — VLM có thể trả
        # thêm field tùy prompt".
        example = {
            "scene": "s",
            "objects": [],
            "actions": [],
            "confidence_note": "model tự thêm field này",
        }
        self.assertTrue(validate_structured_data(example))

    def test_empty_arrays_are_still_valid_minimal_schema(self) -> None:
        # module_caption.md mục 4 (frame prompt): không xác định được -> để rỗng,
        # không bịa số liệu - vẫn phải là structured_data hợp lệ.
        example = {"scene": "không rõ", "objects": [], "actions": []}
        self.assertTrue(validate_structured_data(example))


if __name__ == "__main__":
    unittest.main(verbosity=2)
