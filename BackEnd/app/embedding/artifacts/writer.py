"""Write sharded embedding artifacts."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from BackEnd.app.contracts.embedding import (
    EmbeddingArtifactManifest,
    EmbeddingRecord,
    EmbeddingStatus,
    EntityType,
)
from BackEnd.app.embedding.artifacts.checksum import sha256_file
from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig


class EmbeddingArtifactWriter:
    """Write vectors and trace metadata for one embedding space/run."""

    def __init__(
        self,
        *,
        entity_type: EntityType,
        embedding_space_id: str,
        model_backend: str,
        model_name: str,
        dimension: int,
        run_id: str,
        config: ArtifactWriterConfig = ArtifactWriterConfig(),
    ) -> None:
        self.entity_type = entity_type
        self.embedding_space_id = embedding_space_id
        self.model_backend = model_backend
        self.model_name = model_name
        self.dimension = dimension
        self.run_id = run_id
        self.config = config

    def write(
        self,
        vectors,
        records: list[EmbeddingRecord],
        failures: list[EmbeddingRecord] | None = None,
    ) -> EmbeddingArtifactManifest:
        """Write a single-shard artifact and return its manifest."""

        array = np.asarray(vectors, dtype=self.config.storage_dtype)
        if array.size == 0:
            array = np.empty((0, self.dimension), dtype=self.config.storage_dtype)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (record_count, dimension).")
        success_records = [record for record in records if record.status == EmbeddingStatus.SUCCESS]
        failure_records = list(failures or []) + [
            record for record in records if record.status != EmbeddingStatus.SUCCESS
        ]
        if array.shape[0] != len(success_records):
            raise ValueError("Number of vectors must match success metadata records.")

        final_root = self.config.output_root / self.embedding_space_id / self.run_id
        temp_root = final_root.with_name(f"{final_root.name}.tmp")
        if final_root.exists():
            raise FileExistsError(f"Embedding artifact already exists: {final_root}")
        if temp_root.exists():
            raise FileExistsError(f"Temporary embedding artifact already exists: {temp_root}")

        try:
            manifest = self._write_to_root(
                temp_root,
                array,
                success_records,
                failure_records,
                len(records),
            )
            temp_root.rename(final_root)
            return manifest
        except Exception:
            if temp_root.exists():
                shutil.rmtree(temp_root)
            raise

    def _write_to_root(
        self,
        root: Path,
        array: np.ndarray,
        success_records: list[EmbeddingRecord],
        failure_records: list[EmbeddingRecord],
        record_count: int,
    ) -> EmbeddingArtifactManifest:
        vectors_dir = root / "vectors"
        metadata_dir = root / "metadata"
        failures_dir = root / "failures"
        stats_dir = root / "stats"
        config_dir = root / "config"
        for directory in (vectors_dir, metadata_dir, failures_dir, stats_dir, config_dir):
            directory.mkdir(parents=True, exist_ok=True)

        vector_paths: list[Path] = []
        metadata_paths: list[Path] = []
        shard_size = max(1, int(self.config.rows_per_shard))
        metadata_format = "parquet"
        for shard_index, start in enumerate(range(0, len(success_records), shard_size)):
            end = min(start + shard_size, len(success_records))
            vector_path = vectors_dir / f"part-{shard_index:05d}.npy"
            np.save(vector_path, np.ascontiguousarray(array[start:end]))
            vector_paths.append(vector_path)

            shard_records = [
                replace(
                    record,
                    vector_shard=_relative(root, vector_path),
                    vector_row=row_index,
                )
                for row_index, record in enumerate(success_records[start:end])
            ]
            metadata_path, metadata_format = _write_records(
                metadata_dir / f"part-{shard_index:05d}",
                shard_records,
                preferred_format=metadata_format,
            )
            metadata_paths.append(metadata_path)

        if not success_records:
            vector_path = vectors_dir / "part-00000.npy"
            np.save(vector_path, np.empty((0, self.dimension), dtype=self.config.storage_dtype))
            vector_paths.append(vector_path)
            metadata_path, metadata_format = _write_records(
                metadata_dir / "part-00000",
                [],
            )
            metadata_paths.append(metadata_path)

        failure_path, _ = _write_records(
            failures_dir / "part-00000",
            failure_records,
            preferred_format=metadata_format,
        )
        quality_path = stats_dir / "quality.json"
        quality_path.write_text(
            json.dumps(
                {
                    "record_count": record_count,
                    "success_count": len(success_records),
                    "failure_count": len(failure_records),
                    "dimension": self.dimension,
                    "rows_per_shard": self.config.rows_per_shard,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        resolved_config_path = config_dir / "resolved_config.json"
        resolved_config_path.write_text(
            json.dumps(_jsonable(asdict(self.config)), indent=2),
            encoding="utf-8",
        )

        checksum_paths = vector_paths + metadata_paths + [failure_path, quality_path, resolved_config_path]
        checksums = {_relative(root, path): sha256_file(path) for path in checksum_paths}
        manifest = EmbeddingArtifactManifest(
            artifact_version="embedding-artifact@1.0.0",
            artifact_id=str(uuid4()),
            run_id=self.run_id,
            dataset_id=self.config.dataset_id,
            entity_type=self.entity_type,
            embedding_space_id=self.embedding_space_id,
            model_backend=self.model_backend,
            model_name=self.model_name,
            dimension=self.dimension,
            storage_dtype=self.config.storage_dtype,
            normalized=True,
            record_count=record_count,
            success_count=len(success_records),
            failure_count=len(failure_records),
            shard_count=len(vector_paths),
            vector_shards=tuple(_relative(root, path) for path in vector_paths),
            metadata_shards=tuple(_relative(root, path) for path in metadata_paths),
            failure_shards=(_relative(root, failure_path),),
            checksums=checksums,
            config_hash=checksums[_relative(root, resolved_config_path)],
            created_at=datetime.now(UTC).isoformat(),
            metadata_format=metadata_format,
            sampling_version=_single_metadata_value(success_records, "sampling_version"),
            aggregation_version=_single_metadata_value(success_records, "aggregation_version"),
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(_jsonable(asdict(manifest)), indent=2),
            encoding="utf-8",
        )
        return manifest


def _write_records(
    path_without_suffix: Path,
    records: list[EmbeddingRecord],
    *,
    preferred_format: str | None = None,
) -> tuple[Path, str]:
    record_dicts = [_jsonable(asdict(record)) for record in records]
    if preferred_format in (None, "parquet"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            path = path_without_suffix.with_suffix(".parquet")
            pq.write_table(pa.Table.from_pylist(record_dicts), path)
            return path, "parquet"
        except ImportError:
            pass

    path = path_without_suffix.with_suffix(".jsonl")
    with path.open("w", encoding="utf-8") as file:
        for record in record_dicts:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path, "jsonl"


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _single_metadata_value(records: list[EmbeddingRecord], field_name: str):
    values = {
        getattr(record, field_name)
        for record in records
        if getattr(record, field_name) is not None
    }
    return values.pop() if len(values) == 1 else None


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
