"""Shared contracts for online text, object, and tracking retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetrievalSource = Literal[
    "video_metadata",
    "ocr",
    "transcript",
    "caption",
    "object",
    "object_detection",
    "object_track",
]
EntityType = Literal["video", "shot", "clip", "frame", "track"]


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_range(start_ms: int | None, end_ms: int | None) -> None:
    if start_ms is not None and start_ms < 0:
        raise ValueError("start_ms must be greater than or equal to 0.")
    if end_ms is not None and end_ms < 0:
        raise ValueError("end_ms must be greater than or equal to 0.")
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start_ms must be less than end_ms.")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One normalized candidate emitted by an online retrieval branch."""

    candidate_id: str
    source: RetrievalSource
    entity_type: EntityType
    entity_id: str
    video_id: str
    score: float
    shot_id: str | None = None
    frame_id: str | None = None
    clip_id: str | None = None
    timestamp_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    evidence: str | None = None
    class_id: str | None = None
    class_name: str | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "source", "entity_type", "entity_id", "video_id"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1.")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")
        _validate_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class ObjectConstraint:
    """An object class required by an object/tracking retrieval request."""

    class_name: str
    minimum_confidence: float = 0.0
    minimum_track_duration_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_name", _non_empty(self.class_name, "class_name"))
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        if self.minimum_track_duration_ms < 0:
            raise ValueError("minimum_track_duration_ms must be non-negative.")


@dataclass(frozen=True, slots=True)
class ObjectRetrievalRequest:
    """Filters applied to persisted detections and tracks."""

    objects: tuple[ObjectConstraint, ...]
    top_k: int = 50
    video_ids: tuple[str, ...] = field(default_factory=tuple)
    start_ms: int | None = None
    end_ms: int | None = None
    include_detections: bool = True
    include_tracks: bool = True

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        if not objects:
            raise ValueError("objects must contain at least one constraint.")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "video_ids", tuple(self.video_ids))
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not self.include_detections and not self.include_tracks:
            raise ValueError("At least one object source must be enabled.")
        _validate_range(self.start_ms, self.end_ms)
