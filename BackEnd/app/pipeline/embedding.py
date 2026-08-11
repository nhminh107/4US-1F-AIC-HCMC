"""Embed persisted frames and register their FAISS mappings in PostgreSQL."""

from __future__ import annotations

import numpy as np

from BackEnd import CONFIG
from BackEnd.app.contracts.embedding import ClipRecord, EmbeddingRecord, EmbeddingStatus
from BackEnd.app.contracts.pipeline import (
    ClipEmbeddingMapping,
    ClipWindowMetadata,
    FrameEmbeddingMapping,
    ShotEmbeddingMapping,
    ShotMetadata,
)
from BackEnd.app.database.faiss_db import FAISS_Manager
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.embedding.ImageEmbedding import ImageEmbedder
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder
from BackEnd.app.embedding.clip.service import ClipEmbeddingService
from BackEnd.app.embedding.clip.video_repository import VideoRepository
from BackEnd.app.embedding.model_adapters.clip_vit_b32 import ClipViTB32Adapter
from BackEnd.app.embedding.shot.service import ShotEmbeddingService


class EmbeddingPipeline:
    """Reuse embedding services and one FAISS index across pipeline stages."""

    def __init__(
        self,
        *,
        db: PostgreManager,
        faiss_manager: FAISS_Manager,
        frame_embedder: ImageEmbedder | None = None,
        clip_service: ClipEmbeddingService | None = None,
        shot_service: ShotEmbeddingService | None = None,
        video_repository: VideoRepository | None = None,
    ) -> None:
        self.db = db
        self.faiss_manager = faiss_manager
        self.frame_embedder = frame_embedder
        self.clip_service = clip_service or ClipEmbeddingService(
            decoder=PyAVVideoDecoder(),
            model_adapter=ClipViTB32Adapter(),
        )
        self.shot_service = shot_service or ShotEmbeddingService()
        self.video_repository = video_repository or VideoRepository()
        self._clip_cache: dict[
            str,
            tuple[np.ndarray, list[EmbeddingRecord], list[ClipWindowMetadata]],
        ] = {}

    def embed_frames(self, video_id: str) -> list[FrameEmbeddingMapping]:
        """Embed all persisted frames of one video and save their mappings."""

        if self.frame_embedder is None:
            self.frame_embedder = ImageEmbedder()

        frames = [
            frame
            for frame in self.db.get_frame_record_by_video_id(video_id)
            if frame.source == "extracted"
        ]
        if not frames:
            return []

        existing = self.db.get_frame_embedding_mappings(
            [frame.frame_id for frame in frames],
            index_version=self.faiss_manager.version,
            model_name=self.faiss_manager.model_name,
        )
        self.faiss_manager.validate_ids(
            "frame", [mapping.faiss_id for mapping in existing]
        )
        existing_by_frame = {mapping.frame_id: mapping for mapping in existing}
        pending_frames = [
            frame for frame in frames if frame.frame_id not in existing_by_frame
        ]
        missing_paths = [
            frame.frame_id for frame in pending_frames if frame.frame_path is None
        ]
        if missing_paths:
            raise ValueError(
                "Cannot embed frames without frame_path: "
                f"{missing_paths[:10]}."
            )
        if not pending_frames:
            return [existing_by_frame[frame.frame_id] for frame in frames]

        embeddings = self.frame_embedder.embed_batch(pending_frames)
        new_mappings, _, _ = self.faiss_manager.add_and_save(
            imgs=embeddings,
            imgs_model=pending_frames,
        )
        try:
            self.db.add_frame_embedding_records(new_mappings)
        except Exception:
            self.faiss_manager.rollback(frame_mappings=new_mappings)
            raise

        mappings_by_frame = {
            **existing_by_frame,
            **{mapping.frame_id: mapping for mapping in new_mappings},
        }
        return [mappings_by_frame[frame.frame_id] for frame in frames]

    def embed_clips(self, video_id: str) -> list[ClipEmbeddingMapping]:
        """Embed persisted clip windows of one video and save their mappings."""

        clip_vectors, _, clip_metadata = self._get_clip_embeddings(video_id)
        if not clip_metadata:
            return []

        existing = self.db.get_clip_embedding_mappings(
            [clip.clip_id for clip in clip_metadata],
            index_version=self.faiss_manager.version,
            model_name=self.faiss_manager.model_name,
        )
        self.faiss_manager.validate_ids(
            "clip", [mapping.faiss_id for mapping in existing]
        )
        existing_by_clip = {mapping.clip_id: mapping for mapping in existing}
        pending_positions = [
            index
            for index, clip in enumerate(clip_metadata)
            if clip.clip_id not in existing_by_clip
        ]
        if not pending_positions:
            return [existing_by_clip[clip.clip_id] for clip in clip_metadata]

        pending_vectors = clip_vectors[pending_positions]
        pending_clips = [clip_metadata[index] for index in pending_positions]
        _, new_mappings, _ = self.faiss_manager.add_and_save(
            clips=pending_vectors,
            clips_model=pending_clips,
        )
        try:
            self.db.add_clip_embedding_records(new_mappings)
        except Exception:
            self.faiss_manager.rollback(clip_mappings=new_mappings)
            raise

        mappings_by_clip = {
            **existing_by_clip,
            **{mapping.clip_id: mapping for mapping in new_mappings},
        }
        return [mappings_by_clip[clip.clip_id] for clip in clip_metadata]

    def embed_shot(self, video_id: str) -> list[ShotEmbeddingMapping]:
        """Embed persisted shots of one video by aggregating clip embeddings."""

        shots = self.db.get_list_shot_in_video(video_id)
        if not shots:
            return []

        existing = self.db.get_shot_embedding_mappings(
            [shot.shot_id for shot in shots],
            index_version=self.faiss_manager.version,
            model_name=self.faiss_manager.model_name,
            model_version=self.faiss_manager.model_version,
            pooling_method="mean",
        )
        self.faiss_manager.validate_ids(
            "shot", [mapping.faiss_id for mapping in existing]
        )
        existing_by_shot = {mapping.shot_id: mapping for mapping in existing}
        pending_shots = [
            shot for shot in shots if shot.shot_id not in existing_by_shot
        ]
        if not pending_shots:
            return [existing_by_shot[shot.shot_id] for shot in shots]

        clip_vectors, clip_records, _ = self._get_clip_embeddings(video_id)
        if not clip_records:
            return []

        shot_vectors, records = self.shot_service.aggregate_shots_to_matrix(
            shots=pending_shots,
            clip_records=clip_records,
            clip_vectors=clip_vectors,
        )
        success_records = _successful_records(records, len(shot_vectors))
        shots_by_id = {shot.shot_id: shot for shot in pending_shots}
        mapped_shots = [shots_by_id[record.entity_id] for record in success_records]

        _, _, new_mappings = self.faiss_manager.add_and_save(
            shots=shot_vectors,
            shots_model=mapped_shots,
        )
        try:
            self.db.add_shot_embedding_records(new_mappings)
        except Exception:
            self.faiss_manager.rollback(shot_mappings=new_mappings)
            raise

        mappings_by_shot = {
            **existing_by_shot,
            **{mapping.shot_id: mapping for mapping in new_mappings},
        }
        return [mappings_by_shot[shot.shot_id] for shot in shots]

    def _require_clip_service(self) -> ClipEmbeddingService:
        return self.clip_service

    def _get_clip_embeddings(
        self,
        video_id: str,
    ) -> tuple[np.ndarray, list[EmbeddingRecord], list[ClipWindowMetadata]]:
        cached = self._clip_cache.get(video_id)
        if cached is not None:
            return cached

        embedded = _embed_clip_matrix(
            video_id,
            self.db,
            self._require_clip_service(),
            self.video_repository,
        )
        self._clip_cache[video_id] = embedded
        return embedded


def embed_frames(
    video_id: str,
    pipeline: EmbeddingPipeline,
) -> list[FrameEmbeddingMapping]:
    """Embed and persist all frame vectors for one video."""

    return pipeline.embed_frames(video_id)


def embed_clips(
    video_id: str,
    pipeline: EmbeddingPipeline,
) -> list[ClipEmbeddingMapping]:
    """Embed and persist all clip vectors for one video."""

    return pipeline.embed_clips(video_id)


def embed_shots(
    video_id: str,
    pipeline: EmbeddingPipeline,
) -> list[ShotEmbeddingMapping]:
    """Aggregate and persist shot vectors for one video."""

    return pipeline.embed_shot(video_id)


def _embed_clip_matrix(
    video_id: str,
    db: PostgreManager,
    clip_service: ClipEmbeddingService,
    video_repository: VideoRepository,
) -> tuple[np.ndarray, list[EmbeddingRecord], list[ClipWindowMetadata]]:
    shots, clip_records, clips_by_id = _load_video_clips(video_id, db)
    if not shots or not clip_records:
        return np.empty((0, 0), dtype=np.float32), [], []

    video_asset = video_repository.resolve_video(video_id)
    clip_vectors, records = clip_service.embed_clips_to_matrix(
        clip_records,
        {video_id: video_asset},
    )
    success_records = _successful_records(records, len(clip_vectors))
    return (
        clip_vectors,
        success_records,
        [clips_by_id[record.entity_id] for record in success_records],
    )


def _load_video_clips(
    video_id: str,
    db: PostgreManager,
) -> tuple[list[ShotMetadata], list[ClipRecord], dict[str, ClipWindowMetadata]]:
    shots = db.get_list_shot_in_video(video_id)
    clip_records: list[ClipRecord] = []
    clips_by_id: dict[str, ClipWindowMetadata] = {}

    for shot in shots:
        for clip in db.get_list_clip_in_shot(shot.shot_id):
            clip_records.append(_to_embedding_clip(clip, shot))
            clips_by_id[clip.clip_id] = clip

    return shots, clip_records, clips_by_id


def _to_embedding_clip(
    clip: ClipWindowMetadata,
    shot: ShotMetadata,
) -> ClipRecord:
    scale_type = (
        "full_shot"
        if clip.start_ms == shot.start_ms and clip.end_ms == shot.end_ms
        else "fixed_window"
    )
    return ClipRecord(
        clip_id=clip.clip_id,
        video_id=shot.video_id,
        shot_id=clip.shot_id,
        start_ms=clip.start_ms,
        end_ms=clip.end_ms,
        scale_type=scale_type,
        target_num_frames=CONFIG.CLIP_NUM_FRAMES,
        sampling_strategy=CONFIG.CLIP_SAMPLING_STRATEGY,
        sampling_version=CONFIG.CLIP_SAMPLING_VERSION,
        clip_builder_version=CONFIG.CLIP_BUILDER_VERSION,
        start_frame_idx=clip.start_frame_idx,
        end_frame_idx=clip.end_frame_idx,
    )


def _successful_records(
    records: list[EmbeddingRecord],
    vector_count: int,
) -> list[EmbeddingRecord]:
    failures = [
        record for record in records if record.status != EmbeddingStatus.SUCCESS
    ]
    if failures:
        details = [
            f"{record.entity_id}: {record.status.value}: {record.error_message or 'unknown error'}"
            for record in failures[:5]
        ]
        raise RuntimeError(
            "Embedding stage produced failed records: " + "; ".join(details)
        )

    successful = [
        record for record in records if record.status == EmbeddingStatus.SUCCESS
    ]
    successful.sort(
        key=lambda record: record.vector_row
        if record.vector_row is not None
        else -1
    )
    if len(successful) != vector_count:
        raise RuntimeError(
            "Embedding service returned a different number of successful records "
            "and vectors."
        )
    return successful


if __name__ == "__main__":
    db = PostgreManager()
    faiss_manager = FAISS_Manager(
      img_dim=CONFIG.CLIP_DIMENSION,
      clip_dim=CONFIG.CLIP_DIMENSION,
      shot_dim=CONFIG.CLIP_DIMENSION,
    )

    pipeline = EmbeddingPipeline(
        db=db,
        faiss_manager=faiss_manager,
    )

    pipeline.embed_frames('L21_V005')
