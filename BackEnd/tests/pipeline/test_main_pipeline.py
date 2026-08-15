from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from BackEnd.app.contracts.pipeline import VideoMetadata
from BackEnd.app.pipeline import main_pipeline
from BackEnd.app.pipeline.main_pipeline import Pipeline
from BackEnd.app.pipeline.parallel_runner import WorkerResult


class FakeDatabase:
    def __init__(self, videos: list[VideoMetadata]) -> None:
        self.videos = videos

    def get_list_video(self) -> list[VideoMetadata]:
        return self.videos

    def get_list_shot_in_video(self, video_id: str) -> list[object]:
        return [SimpleNamespace(shot_id=f"{video_id}_S001")]

    def add_tracking_result(self, *, tracks, observations):
        return tracks


class FakeTracker:
    def __init__(self) -> None:
        self.seen_video_ids: list[str] = []
        self.closed = False

    def track_video(self, video: VideoMetadata, shots: list[object]):
        self.seen_video_ids.append(video.video_id)
        return SimpleNamespace(tracks=[], observations=[])

    def close(self) -> None:
        self.closed = True


def _video(video_id: str, path: Path) -> VideoMetadata:
    return VideoMetadata(video_id=video_id, video_path=path)


def test_tracking_model_is_created_only_when_tracking_stage_starts(
    tmp_path: Path,
) -> None:
    database = FakeDatabase(
        [
            _video("L01_V001", tmp_path / "first.mp4"),
            _video("L01_V002", tmp_path / "second.mp4"),
        ]
    )
    tracker = FakeTracker()
    factory_calls = 0

    def build_tracker() -> FakeTracker:
        nonlocal factory_calls
        factory_calls += 1
        return tracker

    pipeline = Pipeline(
        db=database,  # type: ignore[arg-type]
        faiss=SimpleNamespace(),  # type: ignore[arg-type]
        tracker_factory=build_tracker,
    )

    assert factory_calls == 0

    pipeline.run_tracking()

    assert factory_calls == 1
    assert tracker.seen_video_ids == ["L01_V001", "L01_V002"]
    assert tracker.closed is True


def test_embedding_pipeline_is_not_created_until_embedding_stage(tmp_path: Path) -> None:
    database = FakeDatabase([_video("L01_V001", tmp_path / "video.mp4")])
    factory_calls = 0

    class FakeEmbeddingPipeline:
        def embed_frames(self, video_id: str) -> list[object]:
            return []

        def embed_clips(self, video_id: str) -> list[object]:
            return []

        def embed_shot(self, video_id: str) -> list[object]:
            return []

    def build_embedding_pipeline(*args: object) -> FakeEmbeddingPipeline:
        nonlocal factory_calls
        factory_calls += 1
        return FakeEmbeddingPipeline()

    pipeline = Pipeline(
        db=database,  # type: ignore[arg-type]
        faiss=SimpleNamespace(),  # type: ignore[arg-type]
        embedding_pipeline_factory=build_embedding_pipeline,  # type: ignore[arg-type]
    )

    assert factory_calls == 0

    pipeline.run_embeddings()

    assert factory_calls == 1


def test_oom_retry_runs_only_unfinished_videos_in_a_fresh_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = Pipeline(
        db=FakeDatabase([_video("L01_V001", tmp_path / "video.mp4")]),  # type: ignore[arg-type]
        faiss=SimpleNamespace(),  # type: ignore[arg-type]
    )
    retried: list[tuple[str, tuple[str, ...], int]] = []

    def retry_worker(kind: str, video_ids: tuple[str, ...], *, ocr_retry_level: int):
        retried.append((kind, video_ids, ocr_retry_level))
        return WorkerResult(
            kind="enrichment",
            completed_video_ids=video_ids,
            requested_video_ids=video_ids,
        )

    monkeypatch.setattr(main_pipeline, "run_worker_exclusively", retry_worker)
    pipeline._retry_oom_result(
        WorkerResult(
            kind="enrichment",
            completed_video_ids=("L01_V001",),
            oom_video_id="L01_V002",
            oom_message="CUDA out of memory",
            requested_video_ids=("L01_V001", "L01_V002"),
        )
    )

    assert retried == [("enrichment", ("L01_V002",), 1)]
