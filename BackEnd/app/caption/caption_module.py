"""``CaptionGenerator`` — entry point duy nhất của Module Caption.

Cài đặt mục 3, 5 của ``Markdown_Doc/module_caption.md``: sinh caption
(free-text + structured JSON) cho 3 cấp đối tượng Frame/Clip/Shot bằng VLM.
Xem ``Markdown_Doc/caption_pipeline.md`` để có bài viết đầy đủ về pipeline,
các quyết định thiết kế, và vấn đề thực tế gặp phải khi cài đặt.

3 hàm public (``caption_frame``/``caption_clip``/``caption_shot``) độc lập
nhau về mặt chữ ký — mỗi hàm nhận đúng 1 loại entity và trả về
``list[CaptionResult]`` chuẩn contract (module_caption.md mục 3.0), để
``app/pipeline/`` (orchestrator) gọi đúng lúc theo tiến độ pipeline tổng.
``CaptionGenerator`` không biết gì về DB, không tự insert — chỉ trả về data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from BackEnd.app.caption import config
from BackEnd.app.caption.model_client import VLMClient
from BackEnd.app.caption.parsing import parse_caption_output, validate_structured_data
from BackEnd.app.caption.sampling import sample_frames_in_range
from BackEnd.app.contracts.pipeline import (
    CaptionResult,
    ClipWindowMetadata,
    FrameMetadata,
    ShotMetadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PromptSet:
    """3 template prompt đã đọc sẵn từ ``prompts/`` (module_caption.md mục 5)."""

    frame: str
    clip: str
    shot_summarize: str


class CaptionGenerator:
    """Sinh caption cho Frame/Clip/Shot bằng VLM, theo đúng contract ``Caption``."""

    def __init__(
        self,
        vlm_client: VLMClient | None = None,
        *,
        prompts_dir: str | Path = config.PROMPTS_DIR,
    ) -> None:
        """Khởi tạo generator. Model VLM bên trong ``vlm_client`` được nạp lazy.

        Args:
            vlm_client: Client gọi VLM đã cấu hình sẵn (device, 4-bit...). Mặc
                định tự tạo 1 ``VLMClient()`` với cấu hình mặc định trong
                ``config.py``. Cho phép truyền vào từ ngoài chủ yếu để test
                (mock) mà không cần model thật.
            prompts_dir: Thư mục chứa 3 file prompt template. Mặc định
                ``config.PROMPTS_DIR`` (``app/caption/prompts/``).
        """

        self._vlm = vlm_client if vlm_client is not None else VLMClient()
        # model_version phân biệt bản 4-bit/full-precision (module_caption.md
        # mục 6, mục 11) — lấy từ chính client đang dùng, không hard-code.
        self._model_version = (
            config.MODEL_VERSION_4BIT if self._vlm.use_4bit else config.MODEL_VERSION_FULL
        )
        self._prompts = _load_prompts(Path(prompts_dir))

    # ------------------------------------------------------------------
    # 3.1 Frame Caption
    # ------------------------------------------------------------------
    def caption_frame(self, frame: FrameMetadata) -> list[CaptionResult]:
        """Sinh caption cho 1 frame tĩnh (module_caption.md mục 3.1).

        Trả về danh sách rỗng (không raise) nếu không đọc được ảnh nguồn —
        đúng nguyên tắc agent.md mục 9 ("1 bản ghi lỗi/thiếu không được làm
        hỏng cả batch"); lỗi được log kèm ``frame_id`` để dễ truy vết.
        """

        image = _load_image(frame.frame_path, frame.frame_id)
        if image is None:
            return []

        raw_output = self._vlm.generate(self._prompts.frame, images=[image])
        return [self._build_result(raw_output, frame_id=frame.frame_id)]

    # ------------------------------------------------------------------
    # 3.2 Clip Caption
    # ------------------------------------------------------------------
    def caption_clip(
        self, clip: ClipWindowMetadata, sampled_frames: list[FrameMetadata]
    ) -> list[CaptionResult]:
        """Sinh caption cho 1 clip bằng cách feed nhiều frame vào VLM trong 1 lần gọi.

        Hướng "Multi-frame sampling" đã chốt (module_caption.md mục 3.2):
        thay vì dùng model Video-VLM riêng, feed N frame tĩnh trải theo thời
        gian vào 1 lần gọi VLM đa ảnh.

        ``sampled_frames`` được coi là danh sách frame đại diện ĐÃ được sample
        sẵn bởi orchestrator (thường qua ``sampling.sample_frames_in_range``,
        xem ví dụ orchestration ở module_caption.md mục 3.0) — hàm này không
        sample lại từ đầu. Điểm bổ sung duy nhất so với đặc tả: nếu số frame
        truyền vào vượt ``config.MAX_CLIP_SAMPLE_COUNT``, tự áp thêm 1 bước
        downsample an toàn để tránh nạp quá nhiều ảnh vào 1 lần gọi VLM (tốn
        VRAM, kéo dài latency) — xem caption_pipeline.md để biết lý do.
        """

        frames_to_use = _cap_clip_frames(clip, sampled_frames)

        images = _load_images(frames_to_use)
        if not images:
            logger.warning(
                "caption_clip: clip %s không có frame ảnh hợp lệ nào để caption, bỏ qua",
                clip.clip_id,
            )
            return []

        prompt = self._prompts.clip.replace("<NUM_FRAMES>", str(len(images)))
        raw_output = self._vlm.generate(prompt, images=images)
        return [self._build_result(raw_output, clip_id=clip.clip_id)]

    # ------------------------------------------------------------------
    # 3.3 Shot Caption
    # ------------------------------------------------------------------
    def caption_shot(
        self, shot: ShotMetadata, clip_captions: list[CaptionResult]
    ) -> list[CaptionResult]:
        """Tổng hợp caption cấp shot từ các clip caption đã có — KHÔNG chạy lại VLM trên ảnh.

        Đúng nguyên tắc đã chốt ở mục 2.7.3 (Shot Embedding) và nhắc lại ở
        module_caption.md mục 3.3: shot caption được suy ra từ clip caption,
        không caption lại từ đầu bằng ảnh.

        3 nhánh tường minh (agent.md mục 7.9 — "Handle shots without child
        clips using an explicit, documented branch"):

        - 0 clip caption: không có gì để tổng hợp -> KHÔNG bịa caption, trả
          về ``[]`` kèm log cảnh báo. Khác với Shot Embedding (có thể fallback
          bằng vector rỗng có kiểm soát), caption không có giá trị "rỗng hợp
          lệ" để fallback nên không sinh record.
        - 1 clip caption (shot không bị chia nhỏ, hoặc chỉ có đúng 1 clip):
          shot caption = chính clip caption đó, tái sử dụng nguyên caption
          (module_caption.md mục 3.3, case "Shot không chia thành nhiều
          Clip") — không gọi thêm LLM.
        - >= 2 clip caption: nối các clip caption lại, gọi LLM ở chế độ
          TEXT-ONLY (không ảnh) để tóm tắt thành 1 shot caption — dùng lại
          đúng model VLM đã nạp (Qwen2-VL hỗ trợ chat text-only), không nạp
          thêm 1 model LLM riêng (module_caption.md mục 3.3, phương án 1).
        """

        if not clip_captions:
            logger.warning(
                "caption_shot: shot %s không có clip_captions nào để tổng hợp, bỏ qua",
                shot.shot_id,
            )
            return []

        if len(clip_captions) == 1:
            return [_reuse_single_clip_caption(clip_captions[0], shot_id=shot.shot_id)]

        concatenated = "\n".join(f"- {c.caption_text}" for c in clip_captions)
        prompt = self._prompts.shot_summarize.replace("<CLIP_CAPTIONS>", concatenated)
        raw_output = self._vlm.generate(prompt, images=None)
        return [self._build_result(raw_output, shot_id=shot.shot_id)]

    # ------------------------------------------------------------------
    # Helper nội bộ
    # ------------------------------------------------------------------
    def _build_result(
        self,
        raw_output: str,
        *,
        frame_id: str | None = None,
        clip_id: str | None = None,
        shot_id: str | None = None,
    ) -> CaptionResult:
        """Parse output thô của VLM và đóng gói thành 1 ``CaptionResult`` chuẩn contract.

        Không tự set ``confidence`` — ``model.generate()`` (greedy/sampling
        text generation) không cho ra một điểm tin cậy có ý nghĩa trực tiếp
        như OCR/Detection, nên để ``None`` (cột cho phép NULL). Xem hạn chế
        này ở caption_pipeline.md mục "Giới hạn đã biết".
        """

        caption_text, structured_data = parse_caption_output(raw_output)
        if structured_data is not None and not validate_structured_data(structured_data):
            logger.warning(
                "structured_data thiếu field bắt buộc (%s) - vẫn giữ caption_text, "
                "cần rà lại prompt_version=%s",
                ", ".join(("scene", "objects", "actions")),
                config.PROMPT_VERSION,
            )

        return CaptionResult(
            caption_text=caption_text,
            structured_data=structured_data,
            model_name=config.MODEL_NAME,
            model_version=self._model_version,
            prompt_version=config.PROMPT_VERSION,
            confidence=None,
            frame_id=frame_id,
            clip_id=clip_id,
            shot_id=shot_id,
        )


def _reuse_single_clip_caption(source: CaptionResult, *, shot_id: str) -> CaptionResult:
    """Sao chép nội dung 1 clip caption sang shot caption, không gọi lại VLM/LLM.

    Giữ nguyên ``model_name``/``model_version``/``prompt_version``/``confidence``
    của clip caption gốc vì nội dung không đổi — không gán lại metadata của
    ``CaptionGenerator`` hiện tại, tránh đánh lừa là đã chạy lại inference.
    """

    return CaptionResult(
        caption_text=source.caption_text,
        structured_data=source.structured_data,
        model_name=source.model_name,
        model_version=source.model_version,
        prompt_version=source.prompt_version,
        confidence=source.confidence,
        shot_id=shot_id,
    )


def _cap_clip_frames(
    clip: ClipWindowMetadata, sampled_frames: list[FrameMetadata]
) -> list[FrameMetadata]:
    """Chặn số frame nạp vào VLM không vượt quá ``config.MAX_CLIP_SAMPLE_COUNT``."""

    if len(sampled_frames) <= config.MAX_CLIP_SAMPLE_COUNT:
        return sampled_frames
    return sample_frames_in_range(clip, sampled_frames, config.MAX_CLIP_SAMPLE_COUNT)


def _load_image(frame_path: Path | None, entity_id: str) -> Image.Image | None:
    """Đọc 1 ảnh từ đĩa, trả về ``None`` (kèm log rõ lý do) thay vì raise khi lỗi.

    Frame thiếu ảnh cục bộ là tình huống HỢP LỆ trong dự án (agent.md mục
    4.2: metadata bao phủ nhiều video hơn media cục bộ) — không phải lỗi
    nghiêm trọng cần dừng cả batch.
    """

    if frame_path is None:
        logger.warning("Frame %s không có frame_path, bỏ qua khi caption", entity_id)
        return None

    path = Path(frame_path)
    if not path.is_file():
        logger.warning(
            "Frame %s: không tìm thấy file ảnh tại %s, bỏ qua khi caption", entity_id, path
        )
        return None

    try:
        # convert("RGB") để chuẩn hoá mọi ảnh nguồn (kể cả PNG có kênh alpha,
        # ảnh grayscale) về đúng định dạng VLM/processor kỳ vọng.
        return Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning(
            "Frame %s: không đọc được ảnh tại %s (%s), bỏ qua khi caption", entity_id, path, exc
        )
        return None


def _load_images(frames: list[FrameMetadata]) -> list[Image.Image]:
    """Đọc ảnh cho danh sách frame, tự loại bỏ (kèm log) frame nào lỗi thay vì raise."""

    images: list[Image.Image] = []
    for frame in frames:
        image = _load_image(frame.frame_path, frame.frame_id)
        if image is not None:
            images.append(image)
    return images


def _load_prompts(prompts_dir: Path) -> _PromptSet:
    """Đọc 3 file prompt template từ ``prompts_dir`` (module_caption.md mục 5)."""

    return _PromptSet(
        frame=(prompts_dir / "frame_prompt.txt").read_text(encoding="utf-8"),
        clip=(prompts_dir / "clip_prompt.txt").read_text(encoding="utf-8"),
        shot_summarize=(prompts_dir / "shot_summarize_prompt.txt").read_text(encoding="utf-8"),
    )
