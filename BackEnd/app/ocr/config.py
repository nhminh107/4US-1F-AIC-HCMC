"""Hằng số cấu hình cho Module OCR (xem ``Markdown_Doc/module_ocr.md`` mục 4.3, 6, 7, 9).

Cùng tinh thần với ``app/caption/config.py``: mọi giá trị "ma thuật" (đường dẫn
model, ngưỡng confidence, kích thước resize, vùng static_region...) nằm ở một
nơi duy nhất, để ``OCRExtractor``/``MonkeyOCREngine``/``dedup.py`` không phải
hard-code rải rác nhiều nơi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Định danh checkpoint model (module_ocr.md mục 4.3, 6) ---
#
# Đã XÁC NHẬN thật (không còn là giả định best-effort) bằng cách đọc trực
# tiếp README + source code (`parsing/core_runner.py`) của
# github.com/Yuliang-Liu/MonkeyOCRv2:
#   - Checkpoint HuggingFace chính thức: "zenosai/MonkeyOCRv2-S-Parsing"
#     (bản B lớn hơn: "zenosai/MonkeyOCRv2-B-Parsing").
#   - License: Apache License 2.0, cho phép dùng cả nghiên cứu lẫn thương mại.
#   - QUAN TRỌNG: MonkeyOCRv2-Parsing KHÔNG chạy bằng cách nạp thẳng qua
#     ``transformers.AutoModel`` trong tiến trình Python như giả định ban đầu
#     (đúng cho phần "Vision Encoder" của repo, nhưng SAI cho phần
#     "Document Parsing" mà module này cần). Document Parsing được phục vụ
#     qua 1 **vLLM server riêng** (chạy `python serve.py`, cần cài `vllm` +
#     CUDA 12.9+, một tiến trình/máy độc lập) và gọi bằng HTTP theo chuẩn
#     OpenAI Chat Completions (`POST {server_url}/v1/chat/completions`).
#     Xem ``engines/monkeyocr_engine.py`` — engine viết theo đúng kiến trúc
#     HTTP client này, không phải load model local.
DEFAULT_MODEL_REPO_ID = "zenosai/MonkeyOCRv2-S-Parsing"

# Biến môi trường ghi đè checkpoint/tên model — giữ để tương thích đặt tên
# với ``CAPTION_MODEL_PATH`` của Caption module (module_ocr.md mục 6), dùng
# khi cần log/xác định model server đang phục vụ là bản nào.
MODEL_PATH_ENV_VAR = "OCR_MODEL_PATH"

# Tên ngắn gọn dùng khi log/debug — bảng ``OCR`` trong postgre_script.sql
# KHÔNG có cột model_name/model_version (khác với Caption/ObjectDetection),
# nên giá trị này chỉ phục vụ logging nội bộ, không lưu vào OCRResult.
MODEL_NAME = "MonkeyOCRv2-S-Parsing"

# --- Địa chỉ vLLM server phục vụ Document Parsing (BẮT BUỘC phải có 1 tiến
# trình `serve.py` của repo MonkeyOCRv2 đang chạy sẵn ở địa chỉ này — module
# KHÔNG tự khởi động server, chỉ đóng vai trò HTTP client gọi tới). Việc cài
# `vllm` + tải weight + chạy `serve.py` là bước hạ tầng (infra) riêng, thực
# hiện trên máy có GPU thật, ngoài phạm vi code Python của module này.
VLLM_SERVER_URL_ENV_VAR = "OCR_VLLM_SERVER_URL"
DEFAULT_VLLM_SERVER_URL = "http://127.0.0.1:8888"

# Prompt CHÍNH XÁC lấy từ `parsing/core_runner.py` (khoá "END2END" trong
# `ALL_PROMPT`) — yêu cầu model liệt kê toàn bộ block theo thứ tự đọc, kèm
# category + toạ độ + nội dung. Không tự chế prompt khác vì server đã được
# fine-tune để phản hồi đúng format với prompt này.
END2END_PROMPT = (
    "List the document elements in reading order, including their "
    "categories, coordinates, and the content of each element."
)

# Timeout (giây) cho 1 lần gọi HTTP tới vLLM server — document-parsing trên
# 1 ảnh có thể mất vài chục giây tuỳ độ phức tạp, không nên dùng timeout mặc
# định quá ngắn của thư viện requests.
VLLM_REQUEST_TIMEOUT_S = 120


def resolve_model_path() -> str:
    """Trả về tên/định danh checkpoint đang dùng (chỉ để log — xem ghi chú ở trên)."""

    return os.environ.get(MODEL_PATH_ENV_VAR, DEFAULT_MODEL_REPO_ID)


def resolve_vllm_server_url() -> str:
    """Trả về địa chỉ vLLM server sẽ gọi tới, ưu tiên biến môi trường ``OCR_VLLM_SERVER_URL``."""

    return os.environ.get(VLLM_SERVER_URL_ENV_VAR, DEFAULT_VLLM_SERVER_URL)


# --- Tiền xử lý ảnh (module_ocr.md mục 8: "resize cạnh dài về ~960px") ---

# Cạnh dài của ảnh sau khi resize, trước khi đưa vào engine. Giữ nguyên đề
# xuất từ bản thiết kế cũ — ít ảnh hưởng độ chính xác khoanh vùng nhưng giảm
# đáng kể thời gian xử lý.
RESIZE_MAX_SIDE = 960

# Tham số CLAHE (Contrast Limited Adaptive Histogram Equalization) áp dụng
# lên kênh L (độ sáng) của ảnh khi tăng tương phản — dùng cho keyframe tối/mờ.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- Hậu xử lý / lọc kết quả (module_ocr.md mục 5 bước 5, mục 12.7) ---

# Vùng chữ có confidence do engine trả về thấp hơn ngưỡng này bị loại khỏi
# kết quả cuối cùng — đây là 1 trong các tham số cần tinh chỉnh lại sau khi
# test trên keyframe thật (module_ocr.md mục 11, bước 2).
CONFIDENCE_THRESHOLD = 0.5

# 2 vùng có y_min (pixel, trên ảnh đã resize) lệch nhau trong khoảng này được
# coi là cùng 1 "hàng" chữ khi sắp xếp lại theo thứ tự đọc tự nhiên.
LINE_Y_TOLERANCE_PX = 15

# --- pHash dedup theo vùng tĩnh (module_ocr.md mục 7) ---

# Khoảng cách Hamming tối đa giữa 2 perceptual hash để coi là "trùng lặp".
PHASH_HAMMING_THRESHOLD = 4

# Tỉ lệ diện tích giao nhau tối thiểu (so với diện tích vùng chữ do engine
# trả về) để coi 1 vùng chữ là "nằm trong" 1 static_region đã khai báo — dùng
# khi quyết định có áp dụng cache dedup cho vùng đó hay không (dedup.py).
STATIC_REGION_OVERLAP_RATIO = 0.6


@dataclass(frozen=True, slots=True)
class StaticRegion:
    """1 vùng có vị trí cố định trong khung hình (logo đài, khung giờ...).

    Toạ độ normalized [0, 1], cùng hệ quy chiếu với ``OCRResult`` (module_ocr.md
    mục 3). CHỈ khai báo ở đây những vùng thực sự tĩnh về nội dung — banner/
    ticker tin tức chạy chữ (layout cố định nhưng nội dung đổi liên tục) KHÔNG
    được khai báo tại đây, để tránh false-positive dedup đã ghi nhận ở bản
    thiết kế cũ (module_ocr.md mục 7).
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


# Mặc định KHÔNG khai báo sẵn vùng static nào — mỗi kênh/nguồn video có vị trí
# logo/watermark khác nhau, phải khảo sát keyframe thật của đúng nguồn dữ liệu
# đang dùng rồi khai báo tại đây (module_ocr.md mục 11, bước 2). Để trống theo
# mặc định là lựa chọn AN TOÀN: khi chưa khai báo, dedup.py không tự nhận diện
# nhầm bất kỳ vùng nào là tĩnh, mọi vùng chữ đều được engine nhận dạng lại mỗi
# frame (đúng tinh thần mục 8: "pHash dedup chỉ là lớp tối ưu thêm, không phải
# bắt buộc để pipeline chạy đúng").
DEFAULT_STATIC_REGIONS: tuple[StaticRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Gói toàn bộ tham số tinh chỉnh được của module OCR (module_ocr.md mục 5).

    Truyền vào ``OCRExtractor.__init__`` — cho phép override từng tham số khi
    test hoặc khi tinh chỉnh theo bộ keyframe thật, thay vì sửa trực tiếp các
    hằng số module-level ở trên.
    """

    model_path: str = field(default_factory=resolve_model_path)
    resize_max_side: int = RESIZE_MAX_SIDE
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    static_regions: tuple[StaticRegion, ...] = DEFAULT_STATIC_REGIONS
    phash_hamming_threshold: int = PHASH_HAMMING_THRESHOLD
    static_region_overlap_ratio: float = STATIC_REGION_OVERLAP_RATIO
    line_y_tolerance_px: int = LINE_Y_TOLERANCE_PX


# Instance mặc định dùng khi ``OCRExtractor`` không được truyền config riêng.
DEFAULT_CONFIG = OCRConfig()

# Thư mục chứa checkpoint tải về qua huggingface_hub cache (nếu cần override
# thư mục cache mặc định của HF trên máy chạy) — không commit nội dung thư
# mục này vào Git (module_ocr.md mục 6, cùng nguyên tắc với Caption module).
MODEL_CACHE_DIR = Path(os.environ.get("OCR_MODEL_CACHE_DIR", Path.home() / ".cache" / "huggingface"))
