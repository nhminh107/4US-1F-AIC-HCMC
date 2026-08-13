"""Shot embedding service built from validated clip artifacts."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from BackEnd.app.contracts.embedding import (
    ClipRecord,
    EmbeddingArtifactManifest,
    EmbeddingRecord,
)
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd import CONFIG
from BackEnd.CONFIG import SHOT_EMBEDDING_SPACE_ID
from BackEnd.app.embedding.artifacts.reader import (
    load_success_records,
    load_vector_lookup,
    load_vectors,
)
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.common.ids import deterministic_id
from BackEnd.app.embedding.shot.aggregator import aggregate_shot_clips

class ShotEmbeddingService:
    """Aggregate clip embedding artifacts into shot embedding artifacts."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        artifact_writer: EmbeddingArtifactWriter | None = None,
        missing_clip_embedder=None,
    ) -> None:
        self.run_id = run_id or f"shot-run-{uuid4().hex[:12]}"
        self.artifact_writer = artifact_writer
        self.missing_clip_embedder = missing_clip_embedder

    def aggregate_from_clip_artifact(
        self,
        *,
        shots: list[ShotMetadata],
        clip_manifest: EmbeddingArtifactManifest,
        clip_artifact_root,
    ) -> EmbeddingArtifactManifest:
        """Read clip vectors, aggregate by shot, and write a shot artifact."""

        clip_vectors = load_vectors(clip_artifact_root, clip_manifest)
        clip_records = load_success_records(clip_artifact_root, clip_manifest)
        self._validate_clip_artifact_compatibility(clip_manifest, clip_records, clip_vectors)

        records_by_shot: dict[str, list[EmbeddingRecord]] = {}
        vector_lookup = load_vector_lookup(clip_artifact_root, clip_manifest)
        vectors_by_embedding_id = {}
        for record in clip_records:
            records_by_shot.setdefault(record.shot_id or "", []).append(record)
            if record.vector_shard is not None and record.vector_row is not None:
                vectors_by_embedding_id[record.embedding_id] = vector_lookup[
                    (record.vector_shard, record.vector_row)
                ]

        vectors: list[np.ndarray] = []
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingRecord] = []
        for shot in shots:
            source_records = records_by_shot.get(shot.shot_id, [])
            if not source_records:
                if self.missing_clip_embedder is not None:
                    source_records, source_vectors = self.missing_clip_embedder(shot)
                    source_clip_records = [
                        _clip_record_from_embedding_record(record)
                        for record in source_records
                    ]
                else:
                    failures.append(self._failure_record(shot, "no compatible clip embeddings for shot"))
                    continue
            else:
                source_clip_records = [_clip_record_from_embedding_record(record) for record in source_records]
                source_vectors = np.asarray(
                    [vectors_by_embedding_id[record.embedding_id] for record in source_records],
                    dtype=np.float32,
                )

            try:
                shot_vector = aggregate_shot_clips(source_clip_records, source_vectors)
            except ValueError as error:
                failures.append(self._failure_record(shot, str(error)))
                continue

            vector_row = len(vectors)
            vectors.append(shot_vector)
            records.append(
                self._success_record(
                    shot,
                    shot_vector,
                    tuple(record.embedding_id for record in source_records),
                    vector_row,
                )
            )

        writer = self.artifact_writer or EmbeddingArtifactWriter(
            entity_type=CONFIG.EntityType.SHOT,
            embedding_space_id=SHOT_EMBEDDING_SPACE_ID,
            model_backend=clip_manifest.model_backend,
            model_name=clip_manifest.model_name,
            dimension=clip_manifest.dimension,
            run_id=self.run_id,
        )
        import gc
        shot_manifest = writer.write(np.asarray(vectors, dtype=np.float32), records + failures)
        gc.collect()
        return shot_manifest

    def aggregate_shots_to_matrix(
        self,
        *,
        shots: list[ShotMetadata],
        clip_records: list[EmbeddingRecord],
        clip_vectors: np.ndarray,
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        """Aggregate in-memory clip vectors into shot numpy matrix (N_shots, dim) directly on RAM."""

        records_by_shot: dict[str, list[EmbeddingRecord]] = {}
        vectors_by_embedding_id = {}
        for idx, record in enumerate(clip_records):
            records_by_shot.setdefault(record.shot_id or "", []).append(record)
            if idx < len(clip_vectors):
                vectors_by_embedding_id[record.embedding_id] = clip_vectors[idx]

        vectors: list[np.ndarray] = []
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingRecord] = []
        for shot in shots:
            source_records = records_by_shot.get(shot.shot_id, [])
            if not source_records:
                if self.missing_clip_embedder is not None:
                    source_records, source_vectors = self.missing_clip_embedder(shot)
                    source_clip_records = [
                        _clip_record_from_embedding_record(record)
                        for record in source_records
                    ]
                else:
                    failures.append(self._failure_record(shot, "no compatible clip embeddings for shot"))
                    continue
            else:
                source_clip_records = [_clip_record_from_embedding_record(record) for record in source_records]
                source_vectors = np.asarray(
                    [vectors_by_embedding_id[record.embedding_id] for record in source_records],
                    dtype=np.float32,
                )

            try:
                shot_vector = aggregate_shot_clips(source_clip_records, source_vectors)
            except ValueError as error:
                failures.append(self._failure_record(shot, str(error)))
                continue

            vector_row = len(vectors)
            vectors.append(shot_vector)
            records.append(
                self._success_record(
                    shot,
                    shot_vector,
                    tuple(record.embedding_id for record in source_records),
                    vector_row,
                )
            )

        import gc
        vector_matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, clip_vectors.shape[1] if clip_vectors.ndim == 2 else 512), dtype=np.float32)
        )
        gc.collect()
        return vector_matrix, records + failures

    def aggregate_clip_records_to_matrix(
        self,
        *,
        shots: list[ShotMetadata],
        clip_records: list[ClipRecord],
        clip_vectors: np.ndarray,
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        """Aggregate persisted clip vectors without requiring clip artifacts."""

        if len(clip_records) != len(clip_vectors):
            raise ValueError("clip_records and clip_vectors must have the same length.")
        records_by_shot: dict[str, list[tuple[ClipRecord, np.ndarray]]] = {}
        for clip, vector in zip(clip_records, clip_vectors):
            records_by_shot.setdefault(clip.shot_id, []).append((clip, vector))

        vectors: list[np.ndarray] = []
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingRecord] = []
        for shot in shots:
            sources = records_by_shot.get(shot.shot_id, [])
            if not sources:
                failures.append(self._failure_record(shot, "no compatible clip embeddings for shot"))
                continue
            source_clips = [clip for clip, _ in sources]
            source_vectors = np.asarray([vector for _, vector in sources], dtype=np.float32)
            try:
                shot_vector = aggregate_shot_clips(source_clips, source_vectors)
            except ValueError as error:
                failures.append(self._failure_record(shot, str(error)))
                continue
            vector_row = len(vectors)
            vectors.append(shot_vector)
            records.append(
                self._success_record(
                    shot,
                    shot_vector,
                    tuple(clip.clip_id for clip in source_clips),
                    vector_row,
                )
            )

        vector_matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, clip_vectors.shape[1]), dtype=np.float32)
        )
        return vector_matrix, records + failures

    @staticmethod
    def _validate_clip_artifact_compatibility(
        manifest: EmbeddingArtifactManifest,
        records: list[EmbeddingRecord],
        vectors: np.ndarray,
    ) -> None:
        if manifest.entity_type != CONFIG.EntityType.CLIP:
            raise ValueError("Shot aggregation requires a clip embedding artifact.")
        if vectors.ndim != 2 or vectors.shape[1] != manifest.dimension:
            raise ValueError("Clip artifact vector matrix has invalid shape.")
        for record in records:
            if record.embedding_space_id != manifest.embedding_space_id:
                raise ValueError("Clip record embedding space differs from manifest.")
            if record.dimension is not None and record.dimension != manifest.dimension:
                raise ValueError("Clip record dimension differs from manifest.")
            if record.model_backend is not None and record.model_backend != manifest.model_backend:
                raise ValueError("Clip record model backend differs from manifest.")
            if record.model_name is not None and record.model_name != manifest.model_name:
                raise ValueError("Clip record model name differs from manifest.")
            if record.model_revision != manifest.model_revision:
                raise ValueError("Clip record model revision differs from manifest.")
            if record.normalized is not None and record.normalized != manifest.normalized:
                raise ValueError("Clip record normalization differs from manifest.")
            if manifest.sampling_version is not None and record.sampling_version != manifest.sampling_version:
                raise ValueError("Clip record sampling version differs from manifest.")
            if manifest.aggregation_version is not None and record.aggregation_version != manifest.aggregation_version:
                raise ValueError("Clip record aggregation version differs from manifest.")

    def _success_record(
        self,
        shot: ShotMetadata,
        vector: np.ndarray,
        source_embedding_ids: tuple[str, ...],
        vector_row: int,
    ) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=deterministic_id(
                shot.shot_id,
                SHOT_EMBEDDING_SPACE_ID,
                CONFIG.CLIP_MODEL_REVISION,
                CONFIG.SHOT_AGGREGATION_VERSION,
            ),
            embedding_space_id=SHOT_EMBEDDING_SPACE_ID,
            entity_type=CONFIG.EntityType.SHOT,
            entity_id=shot.shot_id,
            video_id=shot.video_id,
            shot_id=shot.shot_id,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            model_revision=CONFIG.CLIP_MODEL_REVISION,
            aggregation_version=CONFIG.SHOT_AGGREGATION_VERSION,
            dimension=int(vector.shape[0]),
            compute_dtype="float32",
            storage_dtype=CONFIG.STORAGE_DTYPE,
            normalized=True,
            vector_norm=float(np.linalg.norm(vector)),
            source_embedding_ids=source_embedding_ids,
            vector_row=vector_row,
            status=CONFIG.EmbeddingStatus.SUCCESS,
            run_id=self.run_id,
        )

    def _failure_record(self, shot: ShotMetadata, error_message: str) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=deterministic_id(
                shot.shot_id,
                SHOT_EMBEDDING_SPACE_ID,
                CONFIG.SHOT_AGGREGATION_VERSION,
                CONFIG.EmbeddingStatus.INVALID_INPUT.value,
            ),
            embedding_space_id=SHOT_EMBEDDING_SPACE_ID,
            entity_type=CONFIG.EntityType.SHOT,
            entity_id=shot.shot_id,
            video_id=shot.video_id,
            shot_id=shot.shot_id,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            status=CONFIG.EmbeddingStatus.INVALID_INPUT,
            run_id=self.run_id,
            error_message=error_message,
        )


def _clip_record_from_embedding_record(record: EmbeddingRecord) -> ClipRecord:
    if record.start_ms is None or record.end_ms is None or record.shot_id is None:
        raise ValueError("Clip embedding record is missing interval or shot_id.")
    return ClipRecord(
        clip_id=record.entity_id,
        video_id=record.video_id,
        shot_id=record.shot_id,
        start_ms=record.start_ms,
        end_ms=record.end_ms,
        scale_type="fixed_window",
        target_num_frames=len(record.sampled_timestamps_ms) or CONFIG.CLIP_NUM_FRAMES,
        sampling_strategy=CONFIG.CLIP_SAMPLING_STRATEGY,
        sampling_version=record.sampling_version or CONFIG.CLIP_SAMPLING_VERSION,
        clip_builder_version=CONFIG.CLIP_BUILDER_VERSION,
    )
