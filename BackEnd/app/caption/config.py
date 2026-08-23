"""Hằng số cấu hình cho Module Caption (xem ``Markdown_Doc/module_caption.md`` mục 5).

Tách riêng khỏi ``caption_module.py`` để mọi giá trị "ma thuật" (tên model,
số frame sample, tham số sinh văn bản...) nằm ở một nơi duy nhất, dễ tra cứu
và chỉnh sửa mà không phải lục qua logic nghiệp vụ của 3 hàm caption chính.
"""

from __future__ import annotations

from pathlib import Path

# --- Định danh model, lưu trực tiếp vào Caption.model_name / model_version /
# prompt_version (module_caption.md mục 2, mục 6, mục 9) ---

# ID repo HuggingFace dùng để tải model + processor thật (xem model_client.py).
MODEL_REPO_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Tên ngắn gọn lưu vào cột Caption.model_name (varchar(100)) — không dùng
# thẳng MODEL_REPO_ID vì cột này chỉ cần định danh model, không cần path đầy
# đủ kèm namespace HuggingFace.
MODEL_NAME = "Qwen2-VL-7B-Instruct"

# 2 bản build cho chất lượng caption khác nhau đáng kể (xem mục 6, mục 11 của
# module_caption.md) nên cần phân biệt tường minh khi tra cứu lại DB — không
# gộp chung một model_version cho cả 2 trường hợp.
MODEL_VERSION_4BIT = "4bit-nf4"
MODEL_VERSION_FULL = "fp16"

# Tăng version này mỗi khi sửa nội dung bất kỳ file nào trong prompts/ — đảm
# bảo so sánh được caption sinh ra trước/sau khi đổi prompt, đúng vai trò của
# cột prompt_version (module_caption.md mục 2).
PROMPT_VERSION = "v1"

# --- Sampling cho Clip Caption (module_caption.md mục 3.2, hướng "Multi-frame
# sampling" đã chốt) ---

# Số frame tối thiểu/tối đa nạp vào 1 lần gọi VLM cho 1 clip. Giới hạn trên
# tồn tại để chặn VRAM/độ trễ tăng vọt nếu orchestrator lỡ truyền vào một
# danh sách frame quá dày (xem caption_module._cap_clip_frames).
MIN_CLIP_SAMPLE_COUNT = 4
MAX_CLIP_SAMPLE_COUNT = 8
DEFAULT_CLIP_SAMPLE_COUNT = 6

# --- Tham số sinh văn bản của VLM ---

# Đủ dài cho 1-4 câu mô tả tiếng Việt kèm khối JSON structured_data phía sau,
# nhưng vẫn chặn trên để tránh model lặp/sinh lan man tốn thời gian.
MAX_NEW_TOKENS = 512

# Quantize 4-bit là cấu hình khuyến nghị để chạy vừa VRAM GPU tầm trung
# (module_caption.md mục 6, mục 7), nhưng đòi hỏi dependency group
# `caption-4bit` trong pyproject.toml. Module mặc định TẮT quantization để có
# thể import/test được trên môi trường pipeline cơ bản; chỉ bật sau khi đã chạy
# `uv sync --group caption-4bit`.
USE_4BIT_DEFAULT = False

# Thư mục chứa 3 file prompt template (module_caption.md mục 5).
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
