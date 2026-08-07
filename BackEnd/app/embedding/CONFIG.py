"""Embedding configuration constants and structured config objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from BackEnd.app.contracts.embedding import EmbeddingStatus, EntityType

try:
    import torch
except ImportError:  # pragma: no cover - only used before ML dependencies are installed.
    torch = None


CLIP_BACKEND = "sentence_transformers"
CLIP_MODEL = "clip-ViT-B-32"
CLIP_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
CLIP_MODEL_REVISION = None
CLIP_DIMENSION = 512

device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None
    else "cpu"
)
batch_size = 64

MIN_BATCH_SIZE = 1
OOM_RETRY = True

CLIP_WINDOW_MS = 10_000
CLIP_STRIDE_MS = 8_000
MIN_NEW_WINDOW_GAP_MS = 2_000
CLIP_BUILDER_VERSION = "clip-builder@1.0.0"

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
    batch_size: int = batch_size
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
