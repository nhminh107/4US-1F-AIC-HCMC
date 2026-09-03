"""Contracts shared by Context and ASR text indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TextSourceType = Literal["frame_context", "asr_segment"]


@dataclass(frozen=True, slots=True)
class TextDocument:
    """One traceable text unit to embed."""

    source_type: TextSourceType
    entity_id: str
    video_id: str
    text: str
    frame_id: str | None = None
    segment_id: str | None = None
    frame_idx: int | None = None
    timestamp_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.source_type not in {"frame_context", "asr_segment"}:
            raise ValueError(f"Unsupported source_type: {self.source_type}")
        if not self.entity_id or not self.video_id or not self.text.strip():
            raise ValueError("entity_id, video_id, and text must not be empty.")
        if self.source_type == "frame_context" and self.frame_id is None:
            raise ValueError("A frame_context document requires frame_id.")
        if self.source_type == "asr_segment" and self.segment_id is None:
            raise ValueError("An asr_segment document requires segment_id.")


@dataclass(frozen=True, slots=True)
class TextIndexManifest:
    """Metadata required to reproduce and validate one text index."""

    artifact_version: str
    build_id: str
    source_type: TextSourceType
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool
    record_count: int
    index_file: str
    mapping_file: str
    checksums: dict[str, str]
    created_at: str
