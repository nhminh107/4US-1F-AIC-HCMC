"""Typed data contracts shared by video-processing pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any, Literal

FrameSource = Literal["official", "extracted"]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata describing a source video."""

    video_id: str
    video_path: Path
    title: str | None = None
    description: str | None = None
    keywords: tuple[str, ...] | None = None
    author: str | None = None
    channel_id: str | None = None
    channel_url: str | None = None
    watch_url: str | None = None
    thumbnail_url: str | None = None
    publish_date: date | None = None
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ShotMetadata:
    """Temporal and frame boundaries of a video shot."""

    shot_id: str
    video_id: str
    shot_index: int
    start_ms: int
    end_ms: int
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Metadata for a keyframe or tracking sample."""

    frame_id: str
    video_id: str
    shot_id: str | None
    timestamp_ms: int
    fps: float
    frame_idx: int
    source: FrameSource | None = None
    n: int | None = None
    pts_time: float | None = None
    frame_path: Path | None = None
    width: int | None = None
    height: int | None = None

@dataclass(frozen=True, slots=True)
class ClassMetadata:
    """A class known by the object-detection pipeline."""

    class_id: str
    class_name: str


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Recognized text and its normalized bounding box."""

    frame_id: str
    n: int
    text: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    language: str | None = None

    def __post_init__(self) -> None:
        """Validate the persisted OCR region contract."""

        if not self.frame_id:
            raise ValueError("OCRResult.frame_id must not be empty.")
        if self.n < 0:
            raise ValueError("OCRResult.n must be non-negative.")
        if not self.text.strip():
            raise ValueError("OCRResult.text must not be empty.")

        coordinates = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("OCRResult coordinates must be finite.")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("OCRResult coordinates must be normalized to [0, 1].")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("OCRResult bounding box must have positive area.")


@dataclass(frozen=True, slots=True)
class ObjectDetectionResult:
    """A normalized object detection produced for a frame."""

    frame_id: str
    class_id: str
    confidence: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    model_name: str | None = None
    model_version: str | None = None
    detection_id: int | None = None


@dataclass(frozen=True, slots=True)
class FrameEmbeddingMapping:
    """FAISS mapping for one frame embedding."""

    faiss_id: int
    index_version: int
    frame_id: str
    model_name: str


@dataclass(frozen=True, slots=True)
class TranscriptSegmentResult:
    """An ASR transcript segment within a video."""

    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectTrackResult:
    """Summary of one object track within a shot."""

    shot_id: str
    class_id: str
    start_ms: int
    end_ms: int
    observation_count: int
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None
    avg_confidence: float | None = None
    tracker_name: str | None = None
    tracker_version: str | None = None
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrackObservationResult:
    """Association between a track and a detection."""

    track_id: int
    detection_id: int


@dataclass(frozen=True, slots=True)
class ClipWindowMetadata:
    """A sampled clip window belonging to a shot."""

    clip_id: str
    shot_id: str
    start_ms: int
    end_ms: int
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None
    sampling_fps: float | None = None
    clip_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ClipEmbeddingMapping:
    """FAISS mapping for one clip embedding."""

    faiss_id: int
    index_version: int
    clip_id: str
    model_name: str


@dataclass(frozen=True, slots=True)
class ShotEmbeddingMapping:
    """FAISS mapping for one pooled shot embedding."""

    faiss_id: int
    index_version: int
    shot_id: str
    model_name: str
    model_version: str
    pooling_method: str = "mean"


@dataclass(frozen=True, slots=True)
class CaptionResult:
    """A caption attached to exactly one frame, clip, or shot."""

    caption_text: str
    model_name: str
    frame_id: str | None = None
    clip_id: str | None = None
    shot_id: str | None = None
    structured_data: dict[str, Any] | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    confidence: float | None = None
    caption_id: int | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        """Require the caption to target exactly one supported entity."""

        target_count = sum(
            target_id is not None
            for target_id in (self.frame_id, self.clip_id, self.shot_id)
        )
        if target_count != 1:
            raise ValueError(
                "CaptionResult must reference exactly one frame, clip, or shot."
            )
