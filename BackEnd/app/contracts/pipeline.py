"""Shared dataclass contracts for the offline video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VideoInput:
    """All local artifacts belonging to one video."""

    video_path: Path
    keyframes_dir: Path
    keyframe_map_path: Path
    embedding_path: Path
    metadata_path: Path | None = None
    objects_dir: Path | None = None
    ocr_path: Path | None = None
    transcript_path: Path | None = None
    captions_path: Path | None = None
    video_id: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Optional real-model stages used by the local orchestrator."""

    run_ocr: bool = False
    run_asr: bool = False
    run_object_detection: bool = False
    run_captioning: bool = False
    recompute_embeddings: bool = False
    allow_model_download: bool = False
    maximum_keyframes: int | None = None
    maximum_audio_seconds: int | None = None
    batch_size: int = 8
    device: str = "auto"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    whisper_model_name: str = "openai/whisper-tiny"
    caption_model_name: str = "Salesforce/blip-image-captioning-base"


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    video_id: str
    video_path: Path
    fps: float
    duration_ms: int
    frame_count: int = 0
    organizer_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneMetadata:
    scene_id: str
    video_id: str
    start_ms: int
    end_ms: int
    scene_index: int = 0
    start_frame_idx: int = 0
    end_frame_idx: int = 0


@dataclass(frozen=True, slots=True)
class SceneBatch:
    video_id: str
    scenes: tuple[SceneMetadata, ...]


@dataclass(frozen=True, slots=True)
class KeyframeMetadata:
    keyframe_id: str
    scene_id: str
    timestamp_ms: int
    image_path: Path
    video_id: str = ""
    frame_idx: int = 0
    source: str = "official"
    is_retrieval: bool = True
    organizer_index: int | None = None


@dataclass(frozen=True, slots=True)
class KeyframeBatch:
    video_id: str
    keyframes: tuple[KeyframeMetadata, ...]


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
    language: str | None = None


@dataclass(frozen=True, slots=True)
class OCRBatch:
    regions: tuple[OCRRegion, ...]


@dataclass(frozen=True, slots=True)
class CaptionResult:
    caption_id: str
    keyframe_id: str
    text: str
    language: str | None = None
    model_name: str | None = None
    model_version: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionBatch:
    captions: tuple[CaptionResult, ...]


@dataclass(frozen=True, slots=True)
class TranscriptSegmentResult:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptBatch:
    video_id: str
    segments: tuple[TranscriptSegmentResult, ...]


@dataclass(frozen=True, slots=True)
class ObjectClassResult:
    class_id: str
    class_name: str


@dataclass(frozen=True, slots=True)
class ObjectDetectionResult:
    detection_id: str
    keyframe_id: str
    class_id: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    source: str = "organizer"
    model_name: str | None = None
    model_version: str | None = None


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    classes: tuple[ObjectClassResult, ...]
    detections: tuple[ObjectDetectionResult, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingMapping:
    faiss_id: int
    keyframe_id: str
    index_version: str
    model_name: str


@dataclass(frozen=True, slots=True)
class FrameEmbeddingBatch:
    keyframe_ids: tuple[str, ...]
    vectors: np.ndarray
    model_name: str


@dataclass(frozen=True, slots=True)
class VectorIndexResult:
    index_name: str
    index_version: str
    model_name: str
    dimension: int
    metric: str
    artifact_path: Path
    checksum: str
    vector_count: int
    mappings: tuple[EmbeddingMapping, ...]


@dataclass(frozen=True, slots=True)
class ClipWindowResult:
    clip_id: str
    video_id: str
    scene_id: str
    start_ms: int
    end_ms: int
    sampling_fps: float


@dataclass(frozen=True, slots=True)
class ClipWindowBatch:
    video_id: str
    windows: tuple[ClipWindowResult, ...]


@dataclass(frozen=True, slots=True)
class SearchDocument:
    document_id: str
    document_type: str
    source_id: str
    text: str
    video_id: str | None = None
    keyframe_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True, slots=True)
class SearchDocumentBatch:
    documents: tuple[SearchDocument, ...]


@dataclass(frozen=True, slots=True)
class SearchArtifactResult:
    artifact_path: Path
    document_count: int
    checksum: str
    backend: str = "jsonl"


@dataclass(frozen=True, slots=True)
class VideoPipelineResult:
    video: VideoMetadata
    scenes: SceneBatch
    keyframes: KeyframeBatch
    ocr: OCRBatch
    transcripts: TranscriptBatch
    detections: DetectionBatch
    captions: CaptionBatch
    embeddings: FrameEmbeddingBatch
    clip_windows: ClipWindowBatch
    search_documents: SearchDocumentBatch


@dataclass(frozen=True, slots=True)
class DatabaseWriteResult:
    database_url: str
    video_count: int
    scene_count: int
    keyframe_count: int
    ocr_count: int
    transcript_count: int
    detection_count: int
    caption_count: int
    clip_window_count: int
    embedding_mapping_count: int


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    is_valid: bool
    video_count: int
    keyframe_count: int
    vector_count: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class OfflineReleaseResult:
    release_id: str
    release_dir: Path
    manifest_path: Path
    manifest_checksum: str
    validation: ValidationReport


@dataclass(frozen=True, slots=True)
class PipelineExecutionResult:
    videos: tuple[VideoPipelineResult, ...]
    frame_index: VectorIndexResult
    search_artifact: SearchArtifactResult
    database_write: DatabaseWriteResult
    release: OfflineReleaseResult
