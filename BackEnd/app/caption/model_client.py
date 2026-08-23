"""Wrapper quanh Qwen2-VL (qua thư viện ``transformers``) cho Module Caption.

Cô lập toàn bộ phần "nặng" — nạp model/processor lên GPU, gọi
``model.generate`` — ra khỏi ``caption_module.py``, để logic 3 cấp caption
(frame/clip/shot) không lẫn với chi tiết inference. Lý do tách được giải
thích đầy đủ ở ``Markdown_Doc/module_caption.md`` mục 5 và
``Markdown_Doc/caption_pipeline.md``.

``VLMClient`` dùng được cho cả 3 trường hợp:
- single-image (Frame Caption)
- multi-image (Clip Caption, hướng "Multi-frame sampling" đã chốt)
- text-only (Shot Caption tổng hợp — không truyền ảnh, dùng lại đúng model
  này ở chế độ chat text-only thay vì nạp thêm 1 LLM riêng)
"""

from __future__ import annotations

import warnings

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from BackEnd.app.caption import config


class VLMClient:
    """Nạp và gọi model Qwen2-VL, dùng chung cho mọi lời gọi caption."""

    def __init__(
        self,
        model_repo_id: str = config.MODEL_REPO_ID,
        *,
        device: str | None = None,
        use_4bit: bool = config.USE_4BIT_DEFAULT,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
    ) -> None:
        """Cấu hình client. Model được nạp lazy (chỉ khi gọi ``generate`` lần đầu).

        Args:
            model_repo_id: ID repo HuggingFace của model VLM.
            device: Chuỗi device của torch (vd ``"cuda"``, ``"cpu"``). Mặc
                định tự chọn CUDA nếu có, cảnh báo rõ ràng nếu phải fallback
                CPU — cùng nguyên tắc với ``ShotExtractor``
                (``shot_extractor.py``), không bao giờ fallback âm thầm.
            use_4bit: Nạp model ở chế độ quantize 4-bit (khuyến nghị production,
                xem module_caption.md mục 6). Yêu cầu cài thêm 'bitsandbytes' +
                'accelerate' — raise RuntimeError kèm hướng dẫn cài đặt nếu
                thiếu, thay vì âm thầm nạp full-precision.
            max_new_tokens: Số token tối đa sinh ra cho mỗi lần gọi.
        """

        self.model_repo_id = model_repo_id
        self.device = _resolve_device(device)
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens

        self._model: Qwen2VLForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None

    def generate(self, prompt_text: str, images: list[Image.Image] | None = None) -> str:
        """Gọi 1 lượt inference: 0..N ảnh + 1 đoạn prompt text -> 1 chuỗi output thô.

        Output thô chưa được parse — việc tách ``caption_text``/``structured_data``
        thuộc về ``parsing.py``, không phải trách nhiệm của client này (module
        này chỉ biết "nói chuyện" với model, không biết ngữ nghĩa output).
        """

        model, processor = self._get_model_and_processor()
        images = images or []

        # Mỗi ảnh tương ứng 1 placeholder {"type": "image"} trong content, xếp
        # trước phần text theo đúng chat template multi-image của Qwen2-VL.
        content: list[dict] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        processor_kwargs: dict = {"text": [chat_text], "return_tensors": "pt", "padding": True}
        if images:
            processor_kwargs["images"] = images

        inputs = processor(**processor_kwargs).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Model.generate() trả về cả phần input đã echo lại ở đầu — cắt bỏ để
        # chỉ giữ token mới sinh ra, tránh caption_text lặp lại nguyên văn prompt.
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs[0].strip()

    def _get_model_and_processor(self):
        """Nạp và cache model + processor khi dùng lần đầu (một lần cho mỗi instance)."""

        if self._model is not None:
            return self._model, self._processor

        model_kwargs: dict = {
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }

        if self.use_4bit:
            model_kwargs["quantization_config"] = _build_4bit_quantization_config()
            # Quantize 4-bit qua bitsandbytes cần device_map thay vì .to(device)
            # thủ công sau khi nạp (bitsandbytes tự dispatch tensor theo map này).
            model_kwargs["device_map"] = {"": 0} if self.device.type == "cuda" else "cpu"

        model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_repo_id, **model_kwargs)
        if not self.use_4bit:
            model.to(self.device)
        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_repo_id)

        self._model = model
        self._processor = processor
        return model, processor


def _build_4bit_quantization_config():
    """Dựng ``BitsAndBytesConfig`` cho chế độ 4-bit, fail rõ ràng nếu thiếu gói.

    Import ``bitsandbytes``/``accelerate`` chỉ khi thật sự cần (use_4bit=True)
    để phần còn lại của module vẫn import/chạy được trên máy chưa cài 2 gói
    này — xem giải thích đầy đủ ở config.USE_4BIT_DEFAULT.
    """

    try:
        from transformers import BitsAndBytesConfig
        import accelerate  # noqa: F401  # chỉ để kiểm tra đã cài, dùng ngầm bởi device_map
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "use_4bit=True yêu cầu 2 gói 'bitsandbytes' và 'accelerate', hiện "
            "chưa được cài trong môi trường mặc định. Cài dependency group "
            "đã khai báo bằng: uv sync --group caption-4bit\n"
            "Hoặc dùng VLMClient(use_4bit=False) để chạy full-precision "
            "(tốn VRAM hơn, xem module_caption.md mục 6/11)."
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def _resolve_device(device: str | None) -> torch.device:
    """Resolve device dùng để inference, luôn cảnh báo (không bao giờ âm thầm) khi fallback về CPU.

    Giữ nguyên logic đã dùng ở ``ShotExtractor._resolve_device`` (xem
    ``app/shot_extractor/shot_extractor.py``) để hành vi nhất quán giữa các
    module chạy model nặng trong dự án.
    """

    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}' but torch.cuda.is_available() is False."
            )
        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")

    warnings.warn(
        "CUDA is not available; VLMClient will run Qwen2-VL on CPU, which is "
        "significantly slower than the GPU execution required by agent.md "
        "section 3. Pass device='cuda' explicitly to fail loudly instead of "
        "falling back.",
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.device("cpu")
