"""Clip embedding service orchestration."""

from __future__ import annotations

from uuid import uuid4

import gc
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from BackEnd.app.contracts.embedding import (
    ClipRecord,
    EmbeddingArtifactManifest,
    EmbeddingRecord,
    VideoAsset,
)
from BackEnd import CONFIG
from BackEnd.CONFIG import CLIP_EMBEDDING_SPACE_ID
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.clip.aggregator import aggregate_clip_frames
from BackEnd.app.embedding.clip.planner import plan_video_work
from BackEnd.app.embedding.clip.sampler import uniform_midpoint_timestamps
from BackEnd.app.embedding.common.ids import deterministic_id
from BackEnd.app.embedding.common.interfaces import ImageTextEmbeddingAdapter, VideoDecoder

class ClipEmbeddingService:
    """Create clip vectors and write a validated clip embedding artifact."""

    def __init__(
        self,
        *,
        decoder: VideoDecoder,
        model_adapter: ImageTextEmbeddingAdapter,
        run_id: str | None = None,
        artifact_writer: EmbeddingArtifactWriter | None = None,
        decode_tolerance_ms: int = CONFIG.DECODE_TOLERANCE_MS,
        num_workers: int = 1,
        enable_gc: bool = True,
    ) -> None:
        self.decoder = decoder
        self.model_adapter = model_adapter
        self.run_id = run_id or f"clip-run-{uuid4().hex[:12]}"
        self.artifact_writer = artifact_writer
        self.decode_tolerance_ms = decode_tolerance_ms
        self.num_workers = max(1, int(num_workers))
        self.enable_gc = enable_gc

    def embed_clips(
        self,
        clips: list[ClipRecord],
        video_assets: dict[str, VideoAsset],
    ) -> EmbeddingArtifactManifest:
        """Embed clips using unique timestamp decode/encode per video."""

        vectors: list[np.ndarray] = []
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingRecord] = []
        dimension = self._resolve_dimension()

        work_units = list(
            plan_video_work(
                clips,
                video_assets,
                max_clips_per_unit=CONFIG.CLIP_MAX_CLIPS_PER_DECODE_UNIT,
            )
        )

        for work_unit, decoded in self._iter_decoded_work_units(work_units):
            video_asset = work_unit.video_asset
            if video_asset is None:
                for clip in work_unit.sorted_clip_records:
                    failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.MEDIA_NOT_FOUND, "video asset not resolved"))
                continue

            assert decoded is not None
            requested_to_vector: dict[int, np.ndarray] = {}
            requested_to_actual: dict[int, int | None] = {}
            valid_images = []
            valid_requested_actual_pairs = []
            unique_actual_timestamps = []
            actual_seen: set[int] = set()
            for requested_timestamp, actual_timestamp, image, status in zip(
                decoded.requested_timestamps_ms,
                decoded.actual_timestamps_ms,
                decoded.images,
                decoded.decode_statuses,
            ):
                if status == "success" and image is not None and actual_timestamp is not None:
                    valid_requested_actual_pairs.append((requested_timestamp, actual_timestamp))
                    if actual_timestamp not in actual_seen:
                        actual_seen.add(actual_timestamp)
                        valid_images.append(image)
                        unique_actual_timestamps.append(actual_timestamp)

            if valid_images:
                try:
                    encoded = self.model_adapter.encode_images(valid_images)
                except Exception as error:  # noqa: BLE001 - preserve adapter boundary context.
                    for clip in work_unit.sorted_clip_records:
                        failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.MODEL_FAILED, str(error)))
                    if self.enable_gc:
                        gc.collect()
                    continue
                actual_to_vector = {
                    int(actual_timestamp): np.asarray(vector, dtype=np.float32)
                    for actual_timestamp, vector in zip(unique_actual_timestamps, encoded)
                }
                for requested_timestamp, actual_timestamp in valid_requested_actual_pairs:
                    requested_to_vector[int(requested_timestamp)] = actual_to_vector[int(actual_timestamp)]
                    requested_to_actual[int(requested_timestamp)] = int(actual_timestamp)

            for clip in work_unit.sorted_clip_records:
                sampled = uniform_midpoint_timestamps(clip)
                frame_vectors = []
                valid_mask = []
                actual_timestamps: list[int | None] = []
                for timestamp in sampled:
                    vector = requested_to_vector.get(timestamp)
                    actual_timestamp = requested_to_actual.get(timestamp)
                    actual_timestamps.append(actual_timestamp)
                    if (
                        vector is None
                        or actual_timestamp is None
                        or not self._actual_timestamp_is_valid(clip, timestamp, actual_timestamp)
                    ):
                        frame_vectors.append(np.zeros(dimension, dtype=np.float32))
                        valid_mask.append(False)
                    else:
                        frame_vectors.append(vector)
                        valid_mask.append(True)
                try:
                    clip_vector = aggregate_clip_frames(np.asarray(frame_vectors), valid_mask)
                except ValueError as error:
                    failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.DECODE_FAILED, str(error)))
                    continue

                vector_row = len(vectors)
                vectors.append(clip_vector)
                records.append(
                    self._success_record(
                        clip,
                        sampled_timestamps=sampled,
                        actual_timestamps=tuple(actual_timestamps),
                        vector=clip_vector,
                        vector_row=vector_row,
                    )
                )

            if self.enable_gc:
                del requested_to_vector
                del requested_to_actual
                del valid_images
                gc.collect()

        all_records = records + failures
        writer = self.artifact_writer or EmbeddingArtifactWriter(
            entity_type=CONFIG.EntityType.CLIP,
            embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=dimension,
            run_id=self.run_id,
        )
        vector_matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, dimension), dtype=np.float32)
        )
        return writer.write(vector_matrix, all_records)

    def embed_clips_to_matrix(
        self,
        clips: list[ClipRecord],
        video_assets: dict[str, VideoAsset],
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        """Embed clips and return raw in-memory numpy matrix (N, dim) and records without requiring disk artifact writing."""

        vectors: list[np.ndarray] = []
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingRecord] = []
        dimension = self._resolve_dimension()

        work_units = list(
            plan_video_work(
                clips,
                video_assets,
                max_clips_per_unit=CONFIG.CLIP_MAX_CLIPS_PER_DECODE_UNIT,
            )
        )

        for work_unit, decoded in self._iter_decoded_work_units(work_units):
            video_asset = work_unit.video_asset
            if video_asset is None:
                for clip in work_unit.sorted_clip_records:
                    failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.MEDIA_NOT_FOUND, "video asset not resolved"))
                continue

            assert decoded is not None
            requested_to_vector: dict[int, np.ndarray] = {}
            requested_to_actual: dict[int, int | None] = {}
            valid_images = []
            valid_requested_actual_pairs = []
            unique_actual_timestamps = []
            actual_seen: set[int] = set()
            for requested_timestamp, actual_timestamp, image, status in zip(
                decoded.requested_timestamps_ms,
                decoded.actual_timestamps_ms,
                decoded.images,
                decoded.decode_statuses,
            ):
                if status == "success" and image is not None and actual_timestamp is not None:
                    valid_requested_actual_pairs.append((requested_timestamp, actual_timestamp))
                    if actual_timestamp not in actual_seen:
                        actual_seen.add(actual_timestamp)
                        valid_images.append(image)
                        unique_actual_timestamps.append(actual_timestamp)

            if valid_images:
                try:
                    encoded = self.model_adapter.encode_images(valid_images)
                except Exception as error:  # noqa: BLE001
                    for clip in work_unit.sorted_clip_records:
                        failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.MODEL_FAILED, str(error)))
                    if self.enable_gc:
                        gc.collect()
                    continue
                actual_to_vector = {
                    int(actual_timestamp): np.asarray(vector, dtype=np.float32)
                    for actual_timestamp, vector in zip(unique_actual_timestamps, encoded)
                }
                for requested_timestamp, actual_timestamp in valid_requested_actual_pairs:
                    requested_to_vector[int(requested_timestamp)] = actual_to_vector[int(actual_timestamp)]
                    requested_to_actual[int(requested_timestamp)] = int(actual_timestamp)

            for clip in work_unit.sorted_clip_records:
                sampled = uniform_midpoint_timestamps(clip)
                frame_vectors = []
                valid_mask = []
                actual_timestamps: list[int | None] = []
                for timestamp in sampled:
                    vector = requested_to_vector.get(timestamp)
                    actual_timestamp = requested_to_actual.get(timestamp)
                    actual_timestamps.append(actual_timestamp)
                    if (
                        vector is None
                        or actual_timestamp is None
                        or not self._actual_timestamp_is_valid(clip, timestamp, actual_timestamp)
                    ):
                        frame_vectors.append(np.zeros(dimension, dtype=np.float32))
                        valid_mask.append(False)
                    else:
                        frame_vectors.append(vector)
                        valid_mask.append(True)
                try:
                    clip_vector = aggregate_clip_frames(np.asarray(frame_vectors), valid_mask)
                except ValueError as error:
                    failures.append(self._failure_record(clip, CONFIG.EmbeddingStatus.DECODE_FAILED, str(error)))
                    continue

                vector_row = len(vectors)
                vectors.append(clip_vector)
                records.append(
                    self._success_record(
                        clip,
                        sampled_timestamps=sampled,
                        actual_timestamps=tuple(actual_timestamps),
                        vector=clip_vector,
                        vector_row=vector_row,
                    )
                )

            if self.enable_gc:
                del requested_to_vector
                del requested_to_actual
                del valid_images
                gc.collect()

        vector_matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, dimension), dtype=np.float32)
        )
        return vector_matrix, records + failures

    def _iter_decoded_work_units(self, work_units):
        """Yield work in order while keeping at most ``num_workers`` decode batches alive.

        Decode can overlap CLIP inference, but completed batches are never
        accumulated for the entire dataset.  The planner's unit cap therefore
        bounds both host RAM and the number of images waiting for the GPU.
        """

        if self.num_workers == 1:
            for work_unit in work_units:
                if work_unit.video_asset is None:
                    yield work_unit, None
                else:
                    yield work_unit, self.decoder.decode_nearest_frames(
                        work_unit.video_asset,
                        work_unit.unique_timestamps_ms,
                    )
            return

        iterator = iter(work_units)
        pending: deque[tuple[object, Future | None]] = deque()

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            try:
                work_unit = next(iterator)
            except StopIteration:
                return False
            if work_unit.video_asset is None:
                pending.append((work_unit, None))
            else:
                pending.append(
                    (
                        work_unit,
                        executor.submit(
                            self.decoder.decode_nearest_frames,
                            work_unit.video_asset,
                            work_unit.unique_timestamps_ms,
                        ),
                    )
                )
            return True

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for _ in range(self.num_workers):
                if not submit_next(executor):
                    break
            while pending:
                work_unit, future = pending.popleft()
                submit_next(executor)
                yield work_unit, None if future is None else future.result()

    def _resolve_dimension(self) -> int:
        if self.artifact_writer is not None:
            return self.artifact_writer.dimension
        if hasattr(self.model_adapter, "metadata"):
            dimension = int(self.model_adapter.metadata().dimension)
            if dimension <= 0:
                raise ValueError("Model adapter metadata returned a non-positive dimension.")
            return dimension
        if hasattr(self.model_adapter, "get_dimension"):
            dimension = int(self.model_adapter.get_dimension())
            if dimension <= 0:
                raise ValueError("Model adapter returned a non-positive dimension.")
            return dimension
        raise ValueError("Cannot resolve embedding dimension from model adapter.")

    def _actual_timestamp_is_valid(
        self,
        clip: ClipRecord,
        requested_timestamp_ms: int,
        actual_timestamp_ms: int,
    ) -> bool:
        return (
            clip.start_ms <= actual_timestamp_ms < clip.end_ms
            and abs(actual_timestamp_ms - requested_timestamp_ms) <= self.decode_tolerance_ms
        )

    def _success_record(
        self,
        clip: ClipRecord,
        *,
        sampled_timestamps: tuple[int, ...],
        actual_timestamps: tuple[int | None, ...],
        vector: np.ndarray,
        vector_row: int,
    ) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=deterministic_id(
                clip.clip_id,
                CLIP_EMBEDDING_SPACE_ID,
                CONFIG.CLIP_MODEL_REVISION,
                clip.sampling_version,
                CONFIG.CLIP_AGGREGATION_VERSION,
            ),
            embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
            entity_type=CONFIG.EntityType.CLIP,
            entity_id=clip.clip_id,
            video_id=clip.video_id,
            shot_id=clip.shot_id,
            start_ms=clip.start_ms,
            end_ms=clip.end_ms,
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            model_revision=CONFIG.CLIP_MODEL_REVISION,
            sampling_version=clip.sampling_version,
            aggregation_version=CONFIG.CLIP_AGGREGATION_VERSION,
            dimension=int(vector.shape[0]),
            compute_dtype="float32",
            storage_dtype=CONFIG.STORAGE_DTYPE,
            normalized=True,
            vector_norm=float(np.linalg.norm(vector)),
            sampled_timestamps_ms=sampled_timestamps,
            actual_timestamps_ms=actual_timestamps,
            vector_row=vector_row,
            status=CONFIG.EmbeddingStatus.SUCCESS,
            run_id=self.run_id,
        )

    def _failure_record(
        self,
        clip: ClipRecord,
        status: CONFIG.EmbeddingStatus,
        error_message: str,
    ) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=deterministic_id(
                clip.clip_id,
                CLIP_EMBEDDING_SPACE_ID,
                CONFIG.CLIP_MODEL_REVISION,
                clip.sampling_version,
                CONFIG.CLIP_AGGREGATION_VERSION,
                status.value,
            ),
            embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
            entity_type=CONFIG.EntityType.CLIP,
            entity_id=clip.clip_id,
            video_id=clip.video_id,
            shot_id=clip.shot_id,
            start_ms=clip.start_ms,
            end_ms=clip.end_ms,
            status=status,
            run_id=self.run_id,
            error_message=error_message,
        )
