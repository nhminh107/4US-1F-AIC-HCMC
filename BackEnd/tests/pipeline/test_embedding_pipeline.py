"""Tests for frame embedding pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from BackEnd.app.contracts.embedding import EmbeddingRecord, EmbeddingStatus, EntityType
from BackEnd.app.contracts.pipeline import (
    ClipEmbeddingMapping,
    ClipWindowMetadata,
    FrameEmbeddingMapping,
    FrameMetadata,
    ShotEmbeddingMapping,
    ShotMetadata,
)
from BackEnd.app.pipeline.embedding import EmbeddingPipeline


class _FakeDatabase:
    def __init__(self, frames: list[FrameMetadata]) -> None:
        self.frames = frames
        self.saved_mappings: list[dict[str, object]] = []
        self.frame_mappings: list[FrameEmbeddingMapping] = []

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        return self.frames

    def get_frame_embedding_mappings(
        self, frame_ids: list[str], **kwargs: object
    ) -> list[FrameEmbeddingMapping]:
        return [
            mapping
            for mapping in self.frame_mappings
            if mapping.frame_id in frame_ids
        ]

    def add_frame_embedding_records(
        self,
        mappings: list[FrameEmbeddingMapping],
    ) -> None:
        self.frame_mappings.extend(mappings)
        self.saved_mappings.extend(
            {
                "faiss_id": mapping.faiss_id,
                "frame_id": mapping.frame_id,
            }
            for mapping in mappings
        )


class _FakeEmbedder:
    def __init__(self) -> None:
        self.received_frames: list[FrameMetadata] = []

    def embed_batch(self, frames: list[FrameMetadata]) -> np.ndarray:
        self.received_frames = frames
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class _FakeFaissManager:
    def __init__(self) -> None:
        self.version = 0
        self.model_name = "test-model"
        self.embeddings: np.ndarray | None = None
        self.frames: list[FrameMetadata] | None = None
        self.add_count = 0
        self.rollback_count = 0

    def validate_ids(self, index_type: str, faiss_ids: list[int]) -> None:
        return None

    def rollback(self, **kwargs: object) -> None:
        self.rollback_count += 1

    def add_and_save(
        self,
        *,
        imgs: np.ndarray,
        imgs_model: list[FrameMetadata],
    ) -> tuple[list[FrameEmbeddingMapping], list[object], list[object]]:
        self.add_count += 1
        self.embeddings = imgs
        self.frames = imgs_model
        return (
            [
                FrameEmbeddingMapping(
                    faiss_id=index + 1,
                    index_version=0,
                    frame_id=frame.frame_id,
                    model_name="test-model",
                )
                for index, frame in enumerate(imgs_model)
            ],
            [],
            [],
        )


def _frame(frame_id: str, frame_path: Path | None = None) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V005",
        shot_id="L21_V005_S000",
        timestamp_ms=0,
        fps=30.0,
        frame_idx=0,
        source="extracted",
        frame_path=frame_path,
    )


def test_embed_frames_persists_faiss_mappings() -> None:
    frames = [
        FrameMetadata(
            frame_id="L21_V005_001",
            video_id="L21_V005",
            shot_id=None,
            timestamp_ms=0,
            fps=30.0,
            frame_idx=0,
            source="official",
            n=1,
        ),
        _frame("L21_V005_E001", Path("data/a.jpg")),
        _frame("L21_V005_E002", Path("data/b.jpg")),
    ]
    database = _FakeDatabase(frames)
    embedder = _FakeEmbedder()
    faiss_manager = _FakeFaissManager()

    pipeline = EmbeddingPipeline(
        db=database,  # type: ignore[arg-type]
        faiss_manager=faiss_manager,  # type: ignore[arg-type]
        frame_embedder=embedder,  # type: ignore[arg-type]
    )
    mappings = pipeline.embed_frames("L21_V005")

    assert [mapping.frame_id for mapping in mappings] == [
        "L21_V005_E001",
        "L21_V005_E002",
    ]
    assert embedder.received_frames == frames[1:]
    assert faiss_manager.frames == frames[1:]
    assert database.saved_mappings[0]["faiss_id"] == 1
    assert database.saved_mappings[1]["frame_id"] == "L21_V005_E002"

    repeated_mappings = pipeline.embed_frames("L21_V005")

    assert repeated_mappings == mappings
    assert faiss_manager.add_count == 1


def test_embed_frames_includes_available_organizer_keyframes(tmp_path: Path) -> None:
    image_path = tmp_path / "official.jpg"
    image_path.write_bytes(b"test-image")
    official = FrameMetadata(
        frame_id="L21_V005_001",
        video_id="L21_V005",
        shot_id=None,
        timestamp_ms=0,
        fps=30.0,
        frame_idx=0,
        source="official",
        n=1,
        frame_path=image_path,
    )
    database = _FakeDatabase([official])
    embedder = _FakeEmbedder()
    faiss_manager = _FakeFaissManager()

    mappings = EmbeddingPipeline(
        db=database,  # type: ignore[arg-type]
        faiss_manager=faiss_manager,  # type: ignore[arg-type]
        frame_embedder=embedder,  # type: ignore[arg-type]
    ).embed_frames("L21_V005")

    assert [mapping.frame_id for mapping in mappings] == [official.frame_id]
    assert embedder.received_frames == [official]


def test_embed_frames_rejects_missing_frame_paths() -> None:
    database = _FakeDatabase([_frame("L21_V005_E001")])

    try:
        EmbeddingPipeline(
            db=database,  # type: ignore[arg-type]
            faiss_manager=_FakeFaissManager(),  # type: ignore[arg-type]
            frame_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        ).embed_frames("L21_V005")
    except ValueError as error:
        assert "frame_path" in str(error)
    else:
        raise AssertionError("Expected missing frame_path validation error.")


def test_embed_frames_rolls_back_faiss_when_database_batch_fails() -> None:
    class _FailingDatabase(_FakeDatabase):
        def add_frame_embedding_records(
            self,
            mappings: list[FrameEmbeddingMapping],
        ) -> None:
            raise RuntimeError("database failed")

    database = _FailingDatabase(
        [_frame("L21_V005_E001", Path("data/a.jpg"))]
    )
    faiss_manager = _FakeFaissManager()
    pipeline = EmbeddingPipeline(
        db=database,  # type: ignore[arg-type]
        faiss_manager=faiss_manager,  # type: ignore[arg-type]
        frame_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="database failed"):
        pipeline.embed_frames("L21_V005")

    assert faiss_manager.rollback_count == 1


class _ClipDatabase:
    def __init__(self) -> None:
        self.shot = ShotMetadata(
            shot_id="L21_V005_S000",
            video_id="L21_V005",
            shot_index=0,
            start_ms=0,
            end_ms=2_000,
            start_frame_idx=0,
            end_frame_idx=59,
        )
        self.clip = ClipWindowMetadata(
            clip_id="L21V005S000C01",
            shot_id=self.shot.shot_id,
            start_ms=0,
            end_ms=2_000,
            start_frame_idx=0,
            end_frame_idx=59,
            sampling_fps=2.0,
        )
        self.saved_clip_mappings: list[dict[str, object]] = []
        self.saved_shot_mappings: list[dict[str, object]] = []
        self.clip_mappings: list[ClipEmbeddingMapping] = []
        self.shot_mappings: list[ShotEmbeddingMapping] = []
        self.shot_mapping_query_kwargs: dict[str, object] = {}

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        return [self.shot]

    def get_list_clip_in_shot(self, shot_id: str) -> list[ClipWindowMetadata]:
        return [self.clip]

    def get_clip_embedding_mappings(
        self, clip_ids: list[str], **kwargs: object
    ) -> list[ClipEmbeddingMapping]:
        return [
            mapping for mapping in self.clip_mappings if mapping.clip_id in clip_ids
        ]

    def get_shot_embedding_mappings(
        self, shot_ids: list[str], **kwargs: object
    ) -> list[ShotEmbeddingMapping]:
        self.shot_mapping_query_kwargs = kwargs
        return [
            mapping for mapping in self.shot_mappings if mapping.shot_id in shot_ids
        ]

    def add_clip_embedding_records(
        self,
        mappings: list[ClipEmbeddingMapping],
    ) -> None:
        self.clip_mappings.extend(mappings)
        self.saved_clip_mappings.extend(
            {
                "clip_id": mapping.clip_id,
                "faiss_id": mapping.faiss_id,
            }
            for mapping in mappings
        )

    def add_shot_embedding_records(
        self,
        mappings: list[ShotEmbeddingMapping],
    ) -> None:
        self.shot_mappings.extend(mappings)
        self.saved_shot_mappings.extend(
            {
                "shot_id": mapping.shot_id,
                "model_version": mapping.model_version,
            }
            for mapping in mappings
        )


class _FakeVideoRepository:
    def resolve_video(self, video_id: str) -> object:
        return object()


class _FakeClipService:
    def __init__(self) -> None:
        self.call_count = 0

    def embed_clips_to_matrix(
        self,
        clips: list[object],
        video_assets: dict[str, object],
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        self.call_count += 1
        clip = clips[0]
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [
                EmbeddingRecord(
                    embedding_id="clip-embedding-1",
                    embedding_space_id="test.clip",
                    entity_type=EntityType.CLIP,
                    entity_id=clip.clip_id,
                    video_id=clip.video_id,
                    shot_id=clip.shot_id,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                    vector_row=0,
                    status=EmbeddingStatus.SUCCESS,
                    run_id="test-run",
                )
            ],
        )


class _FailingClipService(_FakeClipService):
    def embed_clips_to_matrix(
        self,
        clips: list[object],
        video_assets: dict[str, object],
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        self.call_count += 1
        clip = clips[0]
        return (
            np.empty((0, 2), dtype=np.float32),
            [
                EmbeddingRecord(
                    embedding_id="failed-clip-embedding",
                    embedding_space_id="test.clip",
                    entity_type=EntityType.CLIP,
                    entity_id=clip.clip_id,
                    video_id=clip.video_id,
                    shot_id=clip.shot_id,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                    status=EmbeddingStatus.DECODE_FAILED,
                    run_id="test-run",
                    error_message="decode failed",
                )
            ],
        )


class _FakeShotService:
    def aggregate_shots_to_matrix(
        self,
        *,
        shots: list[ShotMetadata],
        clip_records: list[EmbeddingRecord],
        clip_vectors: np.ndarray,
    ) -> tuple[np.ndarray, list[EmbeddingRecord]]:
        shot = shots[0]
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [
                EmbeddingRecord(
                    embedding_id="shot-embedding-1",
                    embedding_space_id="test.shot",
                    entity_type=EntityType.SHOT,
                    entity_id=shot.shot_id,
                    video_id=shot.video_id,
                    shot_id=shot.shot_id,
                    start_ms=shot.start_ms,
                    end_ms=shot.end_ms,
                    vector_row=0,
                    status=EmbeddingStatus.SUCCESS,
                    run_id="test-run",
                )
            ],
        )


class _ClipFaissManager:
    version = 0
    model_name = "test-model"
    model_version = "test-version"

    def __init__(self) -> None:
        self.clip_add_count = 0
        self.shot_add_count = 0

    def validate_ids(self, index_type: str, faiss_ids: list[int]) -> None:
        return None

    def rollback(self, **kwargs: object) -> None:
        return None

    def add_and_save(self, **kwargs: object) -> tuple[list[object], list[object], list[object]]:
        clips = kwargs.get("clips")
        if clips is not None:
            self.clip_add_count += 1
            metadata = kwargs["clips_model"]
            return (
                [],
                [
                    ClipEmbeddingMapping(
                        faiss_id=1,
                        index_version=0,
                        clip_id=metadata[0].clip_id,
                        model_name="test-model",
                    )
                ],
                [],
            )

        self.shot_add_count += 1
        metadata = kwargs["shots_model"]
        return (
            [],
            [],
            [
                ShotEmbeddingMapping(
                    faiss_id=1,
                    index_version=0,
                    shot_id=metadata[0].shot_id,
                    model_name="test-model",
                    model_version="test-version",
                )
            ],
        )


def test_embed_clips_and_shots_persist_faiss_mappings() -> None:
    database = _ClipDatabase()
    clip_service = _FakeClipService()
    faiss_manager = _ClipFaissManager()

    pipeline = EmbeddingPipeline(
        db=database,  # type: ignore[arg-type]
        faiss_manager=faiss_manager,  # type: ignore[arg-type]
        clip_service=clip_service,  # type: ignore[arg-type]
        shot_service=_FakeShotService(),  # type: ignore[arg-type]
        video_repository=_FakeVideoRepository(),  # type: ignore[arg-type]
    )
    clip_mappings = pipeline.embed_clips("L21_V005")
    shot_mappings = pipeline.embed_shot("L21_V005")

    assert [mapping.clip_id for mapping in clip_mappings] == ["L21V005S000C01"]
    assert [mapping.shot_id for mapping in shot_mappings] == ["L21_V005_S000"]
    assert database.saved_clip_mappings[0]["clip_id"] == "L21V005S000C01"
    assert database.saved_shot_mappings[0]["model_version"] == "test-version"
    assert database.shot_mapping_query_kwargs["pooling_method"] == "coverage_weighted_mean"
    assert clip_service.call_count == 1
    assert "L21_V005" not in pipeline._clip_cache

    assert pipeline.embed_clips("L21_V005") == clip_mappings
    assert pipeline.embed_shot("L21_V005") == shot_mappings
    assert faiss_manager.clip_add_count == 1
    assert faiss_manager.shot_add_count == 1


def test_embed_clips_reports_failed_records_before_faiss_write() -> None:
    pipeline = EmbeddingPipeline(
        db=_ClipDatabase(),  # type: ignore[arg-type]
        faiss_manager=_ClipFaissManager(),  # type: ignore[arg-type]
        clip_service=_FailingClipService(),  # type: ignore[arg-type]
        video_repository=_FakeVideoRepository(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="L21V005S000C01.*decode_failed"):
        pipeline.embed_clips("L21_V005")
