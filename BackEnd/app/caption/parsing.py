"""Parse output thô của VLM thành ``(caption_text, structured_data)``.

Prompt (``prompts/*.txt``) luôn yêu cầu model trả lời theo đúng 2 phần: một
đoạn mô tả tự nhiên, theo sau là 1 khối JSON bọc trong code fence
(```` ```json ... ``` ````) — xem ``Markdown_Doc/module_caption.md`` mục 4.
Tuy nhiên VLM không phải lúc nào cũng tuân thủ tuyệt đối định dạng này (quên
code fence, JSON lỗi cú pháp, hoặc không sinh JSON), nên module này là lớp
"làm sạch" độc lập, cô lập rủi ro đó khỏi ``caption_module.py`` (xem lý do
tách file ở module_caption.md mục 5).

Nguyên tắc quan trọng: KHÔNG BAO GIỜ để lỗi parse JSON làm mất luôn
``caption_text`` — free-text vẫn còn giá trị cho search dù ``structured_data``
parse thất bại (structured_data trả về ``None`` trong trường hợp đó).
"""

from __future__ import annotations

import json
import re
from typing import Any

# Khối JSON được bọc trong code fence ```json ... ``` hoặc ``` ... ``` (không
# có tên ngôn ngữ). re.DOTALL để '.' khớp cả xuống dòng bên trong JSON.
_CODE_FENCE_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Fallback khi model quên bọc code fence: bắt cụm {...} cuối cùng trong chuỗi
# (dùng greedy '.*' để ăn tới dấu '}' cuối cùng, tránh cắt cụt JSON lồng nhau).
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)

# Field tối thiểu bắt buộc trong structured_data để đồng bộ giữa các module
# dùng chung (Search/Fusion, Event Extraction) — module_caption.md mục 4.
REQUIRED_STRUCTURED_KEYS: tuple[str, ...] = ("scene", "objects", "actions")


def parse_caption_output(raw_text: str) -> tuple[str, dict[str, Any] | None]:
    """Tách output thô của VLM thành ``(caption_text, structured_data)``.

    Chiến lược tìm JSON theo thứ tự ưu tiên:
    1. Khối JSON trong code fence ``` ```json ... ``` ``` (đúng định dạng đã
       yêu cầu trong prompt).
    2. Nếu không có code fence, thử tìm cụm ``{...}`` trần cuối cùng trong
       chuỗi (model đôi khi quên bọc fence nhưng vẫn sinh đúng JSON).
    3. Nếu JSON tìm được không parse được (``json.JSONDecodeError``) hoặc
       không tìm thấy JSON nào, ``structured_data`` = ``None`` — không raise,
       không làm hỏng ``caption_text``.

    ``caption_text`` là phần văn bản đứng TRƯỚC khối JSON tìm được (đã
    ``strip()``). Nếu không tìm thấy JSON, toàn bộ ``raw_text`` được coi là
    ``caption_text``.
    """

    stripped = raw_text.strip()

    match = _CODE_FENCE_JSON_RE.search(stripped)
    if match is None:
        match = _BARE_JSON_RE.search(stripped)

    if match is None:
        return stripped, None

    caption_text = stripped[: match.start()].strip()
    # Trường hợp hiếm: model chỉ trả JSON, không có mô tả tự nhiên phía trước
    # -> caption_text rỗng. Dùng lại nguyên văn output thô làm caption_text để
    # không trả về chuỗi rỗng cho DB (caption_text NOT NULL trong schema).
    if not caption_text:
        caption_text = stripped

    try:
        structured_data = json.loads(match.group(1))
    except json.JSONDecodeError:
        structured_data = None

    return caption_text, structured_data


def validate_structured_data(structured_data: dict[str, Any] | None) -> bool:
    """Kiểm tra ``structured_data`` có đủ field tối thiểu (mục 4 module_caption.md).

    Chỉ validate SỰ HIỆN DIỆN của 3 field bắt buộc (``scene``, ``objects``,
    ``actions``) — không ép kiểu dữ liệu cụ thể của từng field, vì prompt cho
    phép model tự thêm field khác tuỳ ngữ cảnh (schema không cố định cứng).
    """

    if not isinstance(structured_data, dict):
        return False
    return all(key in structured_data for key in REQUIRED_STRUCTURED_KEYS)
