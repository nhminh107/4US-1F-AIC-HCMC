"""``MonkeyOCREngine`` — engine OCR chính thức duy nhất của module (module_ocr.md mục 4.3, 5).

CẬP NHẬT SAU KHI ĐỌC TRỰC TIẾP SOURCE CODE THẬT của
``github.com/Yuliang-Liu/MonkeyOCRv2`` (``README.md`` + ``parsing/core_runner.py``):
MonkeyOCRv2-Parsing KHÔNG chạy bằng cách nạp checkpoint thẳng vào tiến trình
Python qua ``transformers.AutoModel`` (đó chỉ đúng cho phần "Vision Encoder"
trích feature của repo, không phải phần Document Parsing module này cần).
Document Parsing được phục vụ qua **1 vLLM server riêng** (khởi động bằng
``python serve.py -m <checkpoint> -p 8888`` trên máy có GPU, cần cài gói
``vllm`` + CUDA 12.9+ — một tiến trình/dịch vụ độc lập, KHÔNG thuộc phạm vi
code Python của module này) và được gọi qua HTTP theo chuẩn OpenAI Chat
Completions (``POST {server_url}/v1/chat/completions``).

``MonkeyOCREngine`` vì vậy là 1 **HTTP client** thuần tuý — không nạp model,
không cần GPU/CUDA trong chính tiến trình chạy module này. Mọi việc nặng
(GPU, model) nằm ở phía vLLM server, được vận hành như 1 dịch vụ hạ tầng
riêng (xem checklist triển khai ở ``Markdown_Doc/ocr_pipeline.md``).

Payload/response và cách parse output dưới đây bám sát đúng
``parsing/core_runner.py`` của repo gốc (prompt "END2END", hệ toạ độ bbox
0-1000 quy đổi theo kích thước ảnh, field ``bbox``/``label``/``content``).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import numpy as np
import requests
from PIL import Image

from BackEnd.app.ocr import config
from BackEnd.app.ocr.engines.base import RawTextRegion

logger = logging.getLogger(__name__)

# Model trả về JSON có thể bọc trong code fence Markdown (```json ... ```)
# hoặc có text thừa trước/sau — cần dọn trước khi thử json.loads().
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class MonkeyOCREngine:
    """HTTP client gọi tới vLLM server phục vụ checkpoint MonkeyOCRv2-S-Parsing."""

    def __init__(
        self,
        server_url: str | None = None,
        *,
        model_name: str = config.MODEL_NAME,
        timeout: int = config.VLLM_REQUEST_TIMEOUT_S,
    ) -> None:
        """Cấu hình engine.

        Args:
            server_url: Địa chỉ vLLM server đã chạy sẵn (vd
                ``http://127.0.0.1:8888``). Mặc định lấy từ
                ``config.resolve_vllm_server_url()`` (ưu tiên biến môi trường
                ``OCR_VLLM_SERVER_URL``). Engine KHÔNG tự khởi động server —
                raise lỗi rõ ràng ở lần gọi ``run()`` đầu tiên nếu không kết
                nối được, thay vì fallback âm thầm.
            model_name: Giá trị field ``model`` gửi trong payload OpenAI Chat
                Completions — chỉ để định danh/log phía server, không ảnh
                hưởng việc server đang thực sự phục vụ checkpoint nào.
            timeout: Timeout (giây) cho 1 lần gọi HTTP — document-parsing 1
                ảnh có thể mất vài chục giây, không dùng timeout mặc định
                ngắn của ``requests``.
        """

        self.server_url = (server_url or config.resolve_vllm_server_url()).rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._session = requests.Session()

    def run(self, image: np.ndarray) -> list[RawTextRegion]:
        """Gửi 1 ảnh tới vLLM server, nhận về danh sách block đã parse thành ``RawTextRegion``.

        Trả về ``[]`` nếu ảnh không có chữ (server trả danh sách rỗng) hoặc
        nếu output của server không parse được thành JSON hợp lệ (log warning
        kèm lý do, KHÔNG bịa ra vùng chữ giả — đúng nguyên tắc mục 3).
        """

        if image is None or image.size == 0:
            return []

        image_height, image_width = image.shape[:2]
        # Ảnh đầu vào là numpy BGR (quy ước OpenCV xuyên suốt module), server
        # cần ảnh chuẩn RGB để encode PNG đúng màu.
        pil_image = Image.fromarray(image[:, :, ::-1])

        raw_text = self._chat_completion(pil_image)
        return _parse_end2end_output(raw_text, image_width=image_width, image_height=image_height)

    def run_batch(self, images: list[np.ndarray]) -> list[list[RawTextRegion]]:
        """Chạy ``run()`` cho 1 danh sách ảnh — dùng bởi ``OCRExtractor.extract_batch()``.

        vLLM server tự quản lý batching/scheduling nội bộ giữa nhiều request
        đồng thời (đúng vai trò 1 model-serving engine); ở tầng client này
        chỉ cần gọi tuần tự từng ảnh, KHÔNG tự triển khai lại logic gộp batch
        thủ công — trùng lặp trách nhiệm với chính server.
        """

        return [self.run(image) for image in images]

    def _chat_completion(self, image: Image.Image) -> str:
        """Gọi 1 lượt inference qua endpoint OpenAI Chat Completions của vLLM server."""

        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_to_png_data_uri(image)}},
                        {"type": "text", "text": config.END2END_PROMPT},
                    ],
                }
            ],
        }

        url = f"{self.server_url}/v1/chat/completions"
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"MonkeyOCREngine: không gọi được vLLM server tại {url}. Kiểm tra đã "
                "chạy `python serve.py -m <checkpoint> -p <port>` (repo MonkeyOCRv2) "
                "trên máy phục vụ chưa, hoặc biến môi trường OCR_VLLM_SERVER_URL có "
                f"trỏ đúng địa chỉ không. Lỗi gốc: {exc}"
            ) from exc

        data = response.json()
        return data["choices"][0]["message"]["content"]


def _parse_end2end_output(
    raw_text: str, *, image_width: int, image_height: int
) -> list[RawTextRegion]:
    """Diễn giải output thô (1 chuỗi text chứa JSON list block) thành ``RawTextRegion``.

    Mỗi block hợp lệ theo đúng format server trả về (``parsing/core_runner.py``,
    hàm ``_normalize_model_item``/``parse_end2end_output``) có ``bbox`` (toạ độ
    chuẩn hoá 0-1000 theo TỪNG trục, không phải ``[0, 1]``), ``label`` (loại
    block: Text/Title/Table/Formula/Picture/Caption/...) và ``content`` (text).
    Model không trả confidence/language cho từng block — 2 field này luôn để
    ``None`` trong ``RawTextRegion`` (đúng mục 9: không tự đoán ngôn ngữ khi
    engine không cung cấp).
    """

    items = _extract_json_items(raw_text)
    if not items:
        return []

    regions: list[RawTextRegion] = []
    for item in items:
        normalized = _normalize_item(item)
        if normalized is None:
            continue

        content = normalized["content"].strip()
        if not content:
            # Block có bbox hợp lệ nhưng không có nội dung text (vd 1 khối
            # Picture thuần) - không có gì để OCR, bỏ qua.
            continue

        bbox_px = _map_bbox_to_pixels(normalized["bbox"], image_width, image_height)
        regions.append(RawTextRegion(text=content, bbox=bbox_px, confidence=None, language=None))

    return regions


def _extract_json_items(raw_text: str) -> list[Any]:
    """Parse output thô của server thành 1 list Python, chấp nhận vài định dạng lệch chuẩn.

    Model sinh text tự do nên output có thể bọc trong code fence Markdown
    hoặc lẫn thêm câu dẫn nhập trước/sau JSON — cố gắng phục hồi thay vì
    raise lỗi ngay, nhưng KHÔNG cố "sửa" JSON sai cú pháp thật sự (trả ``[]``
    kèm log warning trong trường hợp đó, không bịa dữ liệu).
    """

    if not raw_text:
        return []

    cleaned = _CODE_FENCE_RE.sub("", raw_text.strip())

    parsed = _try_json_loads(cleaned)
    if parsed is None:
        match = _JSON_ARRAY_RE.search(cleaned)
        parsed = _try_json_loads(match.group(0)) if match else None

    if parsed is None:
        logger.warning(
            "MonkeyOCREngine: không parse được JSON hợp lệ từ output server (%d ký tự), "
            "coi như ảnh không có kết quả",
            len(raw_text),
        )
        return []

    return parsed if isinstance(parsed, list) else []


def _try_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_item(item: Any) -> dict[str, Any] | None:
    """Kiểm tra + chuẩn hoá 1 phần tử JSON thô thành ``{"bbox": [...], "content": str}``.

    Trả về ``None`` (bỏ qua phần tử, không raise) nếu thiếu ``bbox`` hoặc
    ``bbox`` sai định dạng — 1 phần tử hỏng không được làm hỏng các phần tử
    hợp lệ còn lại trong cùng response.
    """

    if not isinstance(item, dict) or "bbox" not in item:
        return None

    bbox = item["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    content = item.get("content", "")
    content = content if isinstance(content, str) else str(content or "")

    return {"bbox": bbox, "content": content}


def _map_bbox_to_pixels(
    bbox_normalized_1000: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Quy đổi bbox từ hệ toạ độ 0-1000 (theo từng trục) của model sang pixel thật.

    Đúng công thức trong ``parsing/core_runner.py._map_bbox_to_image``:
    ``x_pixel = x_1000 / 1000 * width`` (tương tự cho y/height).
    """

    x1, y1, x2, y2 = bbox_normalized_1000
    return (
        x1 / 1000.0 * image_width,
        y1 / 1000.0 * image_height,
        x2 / 1000.0 * image_width,
        y2 / 1000.0 * image_height,
    )


def _image_to_png_data_uri(image: Image.Image) -> str:
    """Encode 1 ``PIL.Image`` thành data URI PNG base64, đúng format field ``image_url``."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
