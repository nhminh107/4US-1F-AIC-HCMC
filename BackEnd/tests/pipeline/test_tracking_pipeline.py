"""Tests for tracking pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from BackEnd.app.contracts.pipeline import (
    FrameMetadata,
    ObjectDetectionResult,
    ObjectTrackResult,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
)
from BackEnd.app.pipeline.tracking import track_video, track_videos
from BackEnd.app.tracking.tracking import TrackingBatchResult


class _FakeDatabase:
    def __init__(self) -> None:
        self.shots = [ShotMetadata("L21_V001_S000", "L21_V001", 0, 0, 1_000)]
        self.frames = [
            FrameMetadata(
                frame_id="L21_V001_E001",
                video_id="L21_V001",
                shot_id="L21_V001_S000",
                timestamp_ms=0,
                fps=25.0,
                frame_idx=0,
                source="extracted",
            )
        ]
        self.detections: list[dict[str, object]] = []
        self.tracks: list[dict[str, object]] = []
        self.observations: list[tuple[int, int]] = []

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        assert video_id == "L21_V001"
        return self.shots

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        assert video_id == "L21_V001"
        return self.frames

    def add_object_detection(self, **kwargs: object) -> SimpleNamespace:
        self.detections.append(kwargs)
        return SimpleNamespace(detection_id=100 + len(self.detections))

    def add_object_track(self, **kwargs: object) -> SimpleNamespace:
        self.tracks.append(kwargs)
        return SimpleNamespace(track_id=200 + len(self.tracks))

    def add_track_observation(
        self,
        track_id: int,
        detection_id: int,
    ) -> SimpleNamespace:
        self.observations.append((track_id, detection_id))
        return SimpleNamespace(track_id=track_id, detection_id=detection_id)


class _FakeTracker:
    def __init__(self, result: TrackingBatchResult) -> None:
        self.result = result
        self.calls: list[tuple[VideoMetadata, list[ShotMetadata], list[FrameMetadata]]] = []

    def track_video(
        self,
        video: VideoMetadata,
        shots: list[ShotMetadata],
        frames: list[FrameMetadata],
    ) -> TrackingBatchResult:
        self.calls.append((video, shots, frames))
        return self.result


def _result() -> TrackingBatchResult:
    persisted_detection = ObjectDetectionResult(
        frame_id="L21_V001_E001",
        class_id="/m/01g317",
        confidence=0.9,
        x_min=0.1,
        x_max=0.8,
        y_min=0.2,
        y_max=0.7,
        detection_id=1,
    )
    temporary_detection = ObjectDetectionResult(
        frame_id="L21_V001T000000",
        class_id=persisted_detection.class_id,
        confidence=0.8,
        x_min=0.1,
        x_max=0.8,
        y_min=0.2,
        y_max=0.7,
        detection_id=2,
    )
    track = ObjectTrackResult(
        shot_id="L21_V001_S000",
        class_id=persisted_detection.class_id,
        start_ms=0,
        end_ms=1,
        observation_count=1,
        track_id=1,
    )
    return TrackingBatchResult(
        detections=[persisted_detection, temporary_detection],
        tracks=[track],
        observations=[
            TrackObservationResult(track_id=1, detection_id=1),
            TrackObservationResult(track_id=1, detection_id=2),
        ],
    )


def test_track_video_persists_tracking_batch() -> None:
    db = _FakeDatabase()
    tracker = _FakeTracker(_result())
    video = VideoMetadata(video_id="L21_V001", video_path=Path("video.mp4"))

    tracks = track_video(video, db, tracker)

    assert tracks == [replace(tracker.result.tracks[0], track_id=201)]
    assert tracker.calls == [(video, db.shots, db.frames)]
    assert len(db.tracks) == 1
    assert len(db.detections) == 1
    assert db.detections[0]["frame_id"] == "L21_V001_E001"
    assert db.observations == [(201, 101)]


def test_track_videos_flattens_tracks() -> None:
    db = _FakeDatabase()
    tracker = _FakeTracker(_result())
    video = VideoMetadata(video_id="L21_V001", video_path=Path("video.mp4"))

    tracks = track_videos([video], db, tracker)

    assert tracks == [replace(tracker.result.tracks[0], track_id=201)]
