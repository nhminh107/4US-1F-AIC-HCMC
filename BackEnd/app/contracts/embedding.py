"""Contracts for offline embedding artifacts and clip/shot embedding stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

ClipScaleType = Literal["full_shot", "fixed_window", "event_window"]
DecodeStatus = Literal["success", "media_not_found", "decode_failed"]


class EntityType(str, Enum):
    FRAME = "frame"
    CLIP = "clip"
    SHOT = "shot"


class EmbeddingStatus(str, Enum):
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    MEDIA_NOT_FOUND = "media_not_found"
    DECODE_FAILED = "decode_failed"
    SAMPLING_FAILED = "sampling_failed"
    MODEL_FAILED = "model_failed"
    INVALID_VECTOR = "invalid_vector"
    WRITE_FAILED = "write_failed"


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_time_range(start_ms: int, end_ms: int) -> None:
    if start_ms < 0:
        raise ValueError("start_ms must be greater than or equal to 0.")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms.")


def _normalize_tuple(value, field_name: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{field_name} must be a tuple or list.")


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """Logical clip window used for clip embedding."""

    clip_id: str
    video_id: str
    shot_id: str
    start_ms: int
    end_ms: int
    scale_type: ClipScaleType
    target_num_frames: int
    sampling_strategy: str
    sampling_version: str
    clip_builder_version: str
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "clip_id", _require_non_empty(self.clip_id, "clip_id"))
        object.__setattr__(self, "video_id", _require_non_empty(self.video_id, "video_id"))
        object.__setattr__(self, "shot_id", _require_non_empty(self.shot_id, "shot_id"))
        _require_time_range(self.start_ms, self.end_ms)
        if self.target_num_frames <= 0:
            raise ValueError("target_num_frames must be greater than 0.")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Model identity persisted with generated embeddings."""

    model_backend: str
    model_name: str
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool = True
    model_weights_hash: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.model_backend, "model_backend")
        _require_non_empty(self.model_name, "model_name")
        _require_non_empty(self.model_id, "model_id")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than 0.")


@dataclass(frozen=True, slots=True)
class VideoAsset:
    """Resolved local video file and basic media metadata."""

    video_id: str
    video_uri: Path
    duration_ms: int | None = None
    container: str | None = None
    codec: str | None = None
    nominal_fps: float | None = None
    has_variable_fps: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _require_non_empty(self.video_id, "video_id"))
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class DecodedFrameBatch:
    """Decoded images for requested timestamps."""

    video_id: str
    images: tuple[object | None, ...]
    requested_timestamps_ms: tuple[int, ...]
    actual_timestamps_ms: tuple[int | None, ...]
    decode_statuses: tuple[DecodeStatus, ...]
    metrics: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.requested_timestamps_ms)
        if not (
            len(self.images) == count
            and len(self.actual_timestamps_ms) == count
            and len(self.decode_statuses) == count
        ):
            raise ValueError("DecodedFrameBatch fields must have matching lengths.")
        if any(timestamp < 0 for timestamp in self.requested_timestamps_ms):
            raise ValueError("requested timestamps must be non-negative.")


@dataclass(frozen=True, slots=True)
class VideoWorkUnit:
    """All clip work planned for one video."""

    video_id: str
    video_asset: VideoAsset | None
    sorted_clip_records: tuple[ClipRecord, ...]
    requested_timestamps_ms: tuple[int, ...]
    unique_timestamps_ms: tuple[int, ...]
    timestamp_to_clip_ids: dict[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        _require_non_empty(self.video_id, "video_id")
        object.__setattr__(
            self,
            "sorted_clip_records",
            _normalize_tuple(self.sorted_clip_records, "sorted_clip_records"),
        )
        object.__setattr__(
            self,
            "requested_timestamps_ms",
            _normalize_tuple(self.requested_timestamps_ms, "requested_timestamps_ms"),
        )
        object.__setattr__(
            self,
            "unique_timestamps_ms",
            _normalize_tuple(self.unique_timestamps_ms, "unique_timestamps_ms"),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """Metadata row paired with one vector row or one failure."""

    embedding_id: str
    embedding_space_id: str
    entity_type: EntityType
    entity_id: str
    video_id: str
    status: EmbeddingStatus
    run_id: str
    shot_id: str | None = None
    timestamp_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    model_backend: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    model_weights_hash: str | None = None
    preprocess_version: str | None = None
    sampling_version: str | None = None
    aggregation_version: str | None = None
    dimension: int | None = None
    compute_dtype: str | None = None
    storage_dtype: str | None = None
    normalized: bool | None = None
    vector_norm: float | None = None
    sampled_timestamps_ms: tuple[int, ...] = field(default_factory=tuple)
    actual_timestamps_ms: tuple[int | None, ...] = field(default_factory=tuple)
    source_embedding_ids: tuple[str, ...] = field(default_factory=tuple)
    vector_shard: str | None = None
    vector_row: int | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.embedding_id, "embedding_id")
        _require_non_empty(self.embedding_space_id, "embedding_space_id")
        _require_non_empty(self.entity_id, "entity_id")
        _require_non_empty(self.video_id, "video_id")
        _require_non_empty(self.run_id, "run_id")
        if self.start_ms is not None and self.end_ms is not None:
            _require_time_range(self.start_ms, self.end_ms)
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors and metadata emitted by an embedding service."""

    vectors: object
    records: tuple[EmbeddingRecord, ...]
    failures: tuple[EmbeddingRecord, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactManifest:
    """Manifest describing one immutable embedding artifact."""

    artifact_version: str
    artifact_id: str
    run_id: str
    dataset_id: str
    entity_type: EntityType
    embedding_space_id: str
    model_backend: str
    model_name: str
    dimension: int
    storage_dtype: str
    normalized: bool
    record_count: int
    success_count: int
    failure_count: int
    shard_count: int
    vector_shards: tuple[str, ...]
    metadata_shards: tuple[str, ...]
    failure_shards: tuple[str, ...]
    checksums: dict[str, str]
    config_hash: str
    created_at: str
    metadata_format: str = "parquet"
    model_revision: str | None = None
    model_weights_hash: str | None = None
    preprocess_version: str | None = None
    sampling_version: str | None = None
    aggregation_version: str | None = None
