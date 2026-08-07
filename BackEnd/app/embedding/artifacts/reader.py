"""Read embedding artifacts written by the offline embedding module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import (
    EmbeddingArtifactManifest,
    EmbeddingRecord,
    EmbeddingStatus,
    EntityType,
)


def load_vectors(artifact_root: Path, manifest: EmbeddingArtifactManifest) -> np.ndarray:
    """Load and concatenate vector shards from an artifact."""

    arrays = [np.load(artifact_root / shard, mmap_mode="r") for shard in manifest.vector_shards]
    if not arrays:
        return np.empty((0, manifest.dimension), dtype=np.float32)
    return np.asarray(np.vstack(arrays), dtype=np.float32)


def load_success_records(
    artifact_root: Path,
    manifest: EmbeddingArtifactManifest,
) -> list[EmbeddingRecord]:
    """Load success metadata rows from an artifact."""

    records: list[EmbeddingRecord] = []
    for shard in manifest.metadata_shards:
        records.extend(_load_record_shard(artifact_root / shard, manifest.metadata_format))
    return records


def load_failure_records(
    artifact_root: Path,
    manifest: EmbeddingArtifactManifest,
) -> list[EmbeddingRecord]:
    """Load failure metadata rows from an artifact."""

    records: list[EmbeddingRecord] = []
    for shard in manifest.failure_shards:
        records.extend(_load_record_shard(artifact_root / shard, manifest.metadata_format))
    return records


def load_vector_lookup(
    artifact_root: Path,
    manifest: EmbeddingArtifactManifest,
) -> dict[tuple[str, int], np.ndarray]:
    """Load vectors keyed by `(relative_shard_path, row_in_shard)`."""

    lookup: dict[tuple[str, int], np.ndarray] = {}
    for shard in manifest.vector_shards:
        matrix = np.load(artifact_root / shard, mmap_mode="r")
        for row_index in range(matrix.shape[0]):
            lookup[(shard, row_index)] = np.asarray(matrix[row_index], dtype=np.float32)
    return lookup


def _load_record_shard(path: Path, metadata_format: str) -> list[EmbeddingRecord]:
    if metadata_format == "parquet" and path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            rows = pq.read_table(path).to_pylist()
        except ImportError as error:
            raise RuntimeError("pyarrow is required to read Parquet artifacts.") from error
    else:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    return [_record_from_dict(row) for row in rows]


def _record_from_dict(row: dict) -> EmbeddingRecord:
    row = dict(row)
    row["entity_type"] = _coerce_entity_type(row["entity_type"])
    row["status"] = _coerce_status(row["status"])
    for field_name in ("sampled_timestamps_ms", "actual_timestamps_ms", "source_embedding_ids"):
        if field_name in row and row[field_name] is not None:
            row[field_name] = tuple(row[field_name])
    return EmbeddingRecord(**row)


def _coerce_entity_type(value) -> EntityType:
    if isinstance(value, EntityType):
        return value
    text = str(value)
    if "." in text:
        text = text.split(".")[-1].lower()
    return EntityType(text)


def _coerce_status(value) -> EmbeddingStatus:
    if isinstance(value, EmbeddingStatus):
        return value
    text = str(value)
    if "." in text:
        text = text.split(".")[-1].lower()
    return EmbeddingStatus(text)
