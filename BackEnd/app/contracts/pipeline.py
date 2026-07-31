from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    video_id: str
    video_path: Path
    fps: float
    duration_ms: int


@dataclass(frozen=True, slots=True)
class SceneMetadata:
    scene_id: str
    video_id: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class KeyframeMetadata:
    keyframe_id: str
    scene_id: str
    timestamp_ms: int
    image_path: Path


@dataclass(frozen=True, slots=True)
class OCRRegion:
    ocr_id: str
    keyframe_id: str
    text: str
    confidence: float | None
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


@dataclass(frozen=True, slots=True)
class CaptionResult:
    caption_id: str
    keyframe_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptSegmentResult:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class EmbeddingMapping:
    faiss_id: int
    keyframe_id: str
    index_version: str
    model_name: str