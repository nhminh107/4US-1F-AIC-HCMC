"""Central configuration for the offline backend pipeline.

Module-specific batch sizes intentionally remain separate because they batch
different workloads. ``GENERAL_BATCH_SIZE`` is only a fallback for new stages
that do not yet need a tuned value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from BackEnd.app.contracts.embedding import EmbeddingStatus, EntityType

try:
    import torch
except ImportError:  # pragma: no cover - supports config-only environments.
    torch = None


# Shared paths and runtime defaults
BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERAL_BATCH_SIZE = 32
# Lowercase alias requested by the pipeline configuration convention.
general_batch_size = GENERAL_BATCH_SIZE


# Video ingestion
VIDEO_METADATA_DIR = DATA_ROOT / "media-info-aic25-b1" / "media-info"
VIDEO_DIR = DATA_ROOT / "video"
VIDEO_DESCRIPTION_MAX_LENGTH = 500


# Audio preprocessing and ASR
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_VAD_FRAME_DURATION_MS = 30
AUDIO_VAD_MODE = 2
AUDIO_PREPROCESSING_OUTPUT_DIR = Path("output/audio_pre")

ASR_MODEL_ID = "khanhld/chunkformer-rnnt-large-vie"
ASR_AUDIO_OUTPUT_DIR = Path("output/asr_audio")
ASR_CHUNK_SIZE = 64
ASR_LEFT_CONTEXT_SIZE = 128
ASR_RIGHT_CONTEXT_SIZE = 128
# ChunkFormer measures this limit in seconds, not number of input records.
ASR_TOTAL_BATCH_DURATION_SECONDS = 1_800
ASR_LANGUAGE = "vi"


# Shot and additional-keyframe extraction
SHOT_BOUNDARY_THRESHOLD = 0.5
SHOT_MIN_DURATION_MS = 500
SHOT_WINDOW_BATCH_SIZE = 8
SHOT_VIDEO_DIR = VIDEO_DIR
SHOT_WEIGHTS_PATH = DATA_ROOT / "models" / "transnetv2-pytorch-weights.pth"

KEYFRAME_OUTPUT_DIR = DATA_ROOT / "keyframes"
KEYFRAME_TARGET_INTERVAL_MS = 2_500
KEYFRAME_MIN_FRAME_GAP = 5
KEYFRAME_MAX_ADDITIONAL_PER_SHOT = 5


# Clip extraction
CLIP_WINDOW_MS = 10_000
CLIP_STRIDE_MS = 8_000
MIN_NEW_WINDOW_GAP_MS = 2_000
CLIP_OUTPUT_ROOT = Path("data/clips")
CLIP_BUILDER_VERSION = "clip-builder@1.0.0"


@dataclass(frozen=True)
class ClipExtractorConfig:
    """Configuration for deterministic overlapping clip windows."""

    split_threshold_ms: int = CLIP_WINDOW_MS
    max_clip_duration_ms: int = CLIP_WINDOW_MS
    stride_ms: int = CLIP_STRIDE_MS
    min_new_window_gap_ms: int = MIN_NEW_WINDOW_GAP_MS
    sampling_fps: float | None = None
    materialize_files: bool = False
    output_root: Path = CLIP_OUTPUT_ROOT
    overwrite: bool = False
    validate_source_duration: bool = True
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "veryfast"
    crf: int = 23

    def __post_init__(self) -> None:
        if self.split_threshold_ms <= 0:
            raise ValueError("split_threshold_ms must be greater than 0")
        if self.max_clip_duration_ms <= 0:
            raise ValueError("max_clip_duration_ms must be greater than 0")
        if self.split_threshold_ms > self.max_clip_duration_ms:
            raise ValueError(
                "split_threshold_ms must be less than or equal to "
                "max_clip_duration_ms"
            )
        if self.stride_ms <= 0:
            raise ValueError("stride_ms must be greater than 0")
        if self.stride_ms > self.max_clip_duration_ms:
            raise ValueError(
                "stride_ms must be less than or equal to max_clip_duration_ms"
            )
        if self.min_new_window_gap_ms <= 0:
            raise ValueError("min_new_window_gap_ms must be greater than 0")
        if self.min_new_window_gap_ms > self.stride_ms:
            raise ValueError(
                "min_new_window_gap_ms must be less than or equal to stride_ms"
            )
        if self.sampling_fps is not None and self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be greater than 0")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")


# OCR
DEFAULT_VIETOCR_CONFIG_PATH = (
    BACKEND_ROOT / "app" / "ocr" / "configs" / "vietocr_vgg_transformer.yml"
)
OCR_DETECTION_BATCH_SIZE = 4
OCR_RECOGNITION_BATCH_SIZE = GENERAL_BATCH_SIZE
OCR_LEGACY_CHUNK_SIZE = 100
OCR_LEGACY_DETECTION_CONFIDENCE_THRESHOLD = 0.65
OCR_DEDUP_HAMMING_DISTANCE_THRESHOLD = 4


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Reproducible OCR model and inference settings."""

    detection_model_name: str = "PP-OCRv5_mobile_det"
    recognition_backend: str = "vietocr"
    recognition_model_name: str = "vgg_transformer"
    detection_model_dir: Path | None = None
    recognition_model_dir: Path | None = None
    recognition_config_path: Path | None = DEFAULT_VIETOCR_CONFIG_PATH
    model_version: str = "paddleocr-3.x"
    language: str = "vi"
    device: str | None = None
    engine: str | None = None
    enable_hpi: bool = False
    disable_new_ir: bool = True
    precision: str = "fp32"
    cpu_threads: int = 8
    detection_batch_size: int = OCR_DETECTION_BATCH_SIZE
    recognition_batch_size: int = OCR_RECOGNITION_BATCH_SIZE
    detection_limit_side_len: int = 1280
    detection_pixel_threshold: float = 0.3
    detection_box_threshold: float = 0.5
    detection_unclip_ratio: float = 1.6
    recognition_score_threshold: float = 0.45
    crop_padding_ratio: float = 0.04
    minimum_crop_side: int = 3

    def __post_init__(self) -> None:
        if not self.detection_model_name or not self.recognition_model_name:
            raise ValueError("OCR model names must not be empty.")
        if self.recognition_backend not in {"paddleocr", "vietocr"}:
            raise ValueError(
                "recognition_backend must be either 'paddleocr' or 'vietocr'."
            )
        if not self.language:
            raise ValueError("OCR language must not be empty.")
        if self.detection_batch_size <= 0 or self.recognition_batch_size <= 0:
            raise ValueError("OCR batch sizes must be positive.")
        if self.detection_limit_side_len <= 0:
            raise ValueError("detection_limit_side_len must be positive.")
        if self.minimum_crop_side <= 0:
            raise ValueError("minimum_crop_side must be positive.")
        if self.crop_padding_ratio < 0:
            raise ValueError("crop_padding_ratio must be non-negative.")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive.")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be either 'fp32' or 'fp16'.")

        thresholds = {
            "detection_pixel_threshold": self.detection_pixel_threshold,
            "detection_box_threshold": self.detection_box_threshold,
            "recognition_score_threshold": self.recognition_score_threshold,
        }
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")


# Object detection
OBJECT_DETECTION_KEYFRAMES_DIR = KEYFRAME_OUTPUT_DIR
OBJECT_DETECTION_OUTPUT_PATH = (
    BACKEND_ROOT
    / "app"
    / "object_detection"
    / "output"
    / "object_detection_results.json"
)
OBJECT_DETECTION_CHUNK_SIZE = 100
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.25
OBJECT_DETECTION_NMS_IOU_THRESHOLD = 0.45
OBJECT_DETECTION_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OPENIMAGES_MODEL_URL = (
    "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
)
OPENIMAGES_MODEL_NAME = "faster_rcnn/inception_resnet_v2"
OPENIMAGES_MODEL_VERSION = "openimages_v4/1"


# Search indexing
ELASTICSEARCH_BULK_BATCH_SIZE = 500
ELASTICSEARCH_INDEX_SCHEMA_VERSION = "text-index-schema@1.0.0"


# Tracking
@dataclass(frozen=True, slots=True)
class TrackingConfig:
    sampling_fps: float = 2.0
    track_activation_threshold: float = 0.25
    high_confidence_threshold: float = 0.35
    minimum_iou_threshold: float = 0.20
    lost_track_buffer: int = 30

    def __post_init__(self) -> None:
        if self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be positive.")


# Embedding
CLIP_BACKEND = "sentence_transformers"
CLIP_MODEL = "clip-ViT-B-32"
CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
CLIP_MODEL_REVISION = None
CLIP_DIMENSION = 512
CLIP_EMBEDDING_SPACE_ID = "clip.clip_vit_b32.masked_mean16_v1"
SHOT_EMBEDDING_SPACE_ID = "shot.clip_vit_b32.coverage_pool_v1"

device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None
    else "cpu"
)
EMBEDDING_BATCH_SIZE = 64
# Backward-compatible lowercase name used by the current embedding module.
batch_size = EMBEDDING_BATCH_SIZE

MIN_BATCH_SIZE = 1
OOM_RETRY = True
CLIP_NUM_FRAMES = 16
CLIP_SAMPLING_STRATEGY = "uniform_midpoint"
CLIP_SAMPLING_VERSION = "clip-sampling@1.0.0"
DECODE_TOLERANCE_MS = 500
DECODE_SEEK_GROUP_GAP_MS = 2_000
CLIP_AGGREGATION = "masked_mean"
CLIP_AGGREGATION_VERSION = "clip-aggregation@1.0.0"
SHOT_AGGREGATION = "coverage_weighted_mean"
SHOT_AGGREGATION_VERSION = "shot-aggregation@1.0.0"
NORMALIZE_EMBEDDINGS = True
STORAGE_DTYPE = "float32"
ROWS_PER_SHARD = 25_000
OUTPUT_ROOT = "artifacts/embeddings"


@dataclass(frozen=True, slots=True)
class ClipBuilderConfig:
    dataset_id: str = "aic_hcm2026"
    window_ms: int = CLIP_WINDOW_MS
    stride_ms: int = CLIP_STRIDE_MS
    min_new_window_gap_ms: int = MIN_NEW_WINDOW_GAP_MS
    target_num_frames: int = CLIP_NUM_FRAMES
    sampling_strategy: str = CLIP_SAMPLING_STRATEGY
    sampling_version: str = CLIP_SAMPLING_VERSION
    clip_builder_version: str = CLIP_BUILDER_VERSION


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeConfig:
    clip_backend: str = CLIP_BACKEND
    clip_model: str = CLIP_MODEL
    clip_model_id: str = CLIP_MODEL_ID
    clip_model_revision: str | None = CLIP_MODEL_REVISION
    clip_dimension: int = CLIP_DIMENSION
    batch_size: int = EMBEDDING_BATCH_SIZE
    min_batch_size: int = MIN_BATCH_SIZE
    oom_retry: bool = OOM_RETRY
    normalize_embeddings: bool = NORMALIZE_EMBEDDINGS
    storage_dtype: str = STORAGE_DTYPE
    output_root: Path = Path(OUTPUT_ROOT)
    decode_tolerance_ms: int = DECODE_TOLERANCE_MS


@dataclass(frozen=True, slots=True)
class ArtifactWriterConfig:
    dataset_id: str = "aic_hcm2026"
    output_root: Path = Path(OUTPUT_ROOT)
    rows_per_shard: int = ROWS_PER_SHARD
    storage_dtype: str = STORAGE_DTYPE
