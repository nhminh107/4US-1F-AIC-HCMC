"""Data contracts for FrameContext V1."""

from __future__ import annotations

from dataclasses import dataclass, field


FRAME_CONTEXT_SCHEMA_VERSION = "frame-context@1.0.0"


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """Specialist evidence collected for one canonical frame."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    captions: tuple[str, ...] = field(default_factory=tuple)
    ocr_texts: tuple[str, ...] = field(default_factory=tuple)
    object_labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.frame_id or not self.video_id:
            raise ValueError("frame_id and video_id must not be empty.")
        if self.frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("frame_idx and timestamp_ms must be non-negative.")


@dataclass(frozen=True, slots=True)
class FrameContextRecord:
    """Deterministic text representation derived from frame evidence."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    context_text: str
    caption_text: str
    ocr_text: str
    object_text: str
    schema_version: str = FRAME_CONTEXT_SCHEMA_VERSION
