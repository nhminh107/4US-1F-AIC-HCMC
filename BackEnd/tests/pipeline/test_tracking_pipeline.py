"""Tests for independent tracking pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from BackEnd.app.contracts.pipeline import (
    ObjectTrackResult,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
)
from BackEnd.app.pipeline.tracking import track_video
from BackEnd.app.tracking.tracking import TrackingBatchResult


class _FakeDatabase:
    def __init__(self) -> None:
        self.shots = [ShotMetadata("L21_V001_S000", "L21_V001", 0, 0, 1_000)]
        self.persisted_batches: list[
            tuple[list[ObjectTrackResult], list[TrackObservationResult]]
        ] = []

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        assert video_id == "L21_V001"
        return self.shots

    def add_tracking_result(
        self,
        tracks: list[ObjectTrackResult],
        observations: list[TrackObservationResult],
    ) -> list[ObjectTrackResult]:
        self.persisted_batches.append((tracks, observations))
        return [replace(track, track_id=201) for track in tracks]


class _FakeTracker:
    def __init__(self, result: TrackingBatchResult) -> None:
        self.result = result
        self.calls: list[tuple[VideoMetadata, list[ShotMetadata]]] = []

    def track_video(
        self,
        video: VideoMetadata,
        shots: list[ShotMetadata],
    ) -> TrackingBatchResult:
        self.calls.append((video, shots))
        return self.result


def _result() -> TrackingBatchResult:
    track = ObjectTrackResult(
        shot_id="L21_V001_S000",
        class_id="/m/01g317",
        start_ms=0,
        end_ms=1,
        observation_count=1,
        model_name="YOLO26",
        model_version="yolo26n.pt",
        tracker_name="ByteTrack",
        tracker_version="8.4.116",
        sampling_fps=2.0,
        mapping_version="coco80-openimages-v1",
        start_frame_idx=0,
        end_frame_idx=0,
        avg_confidence=0.9,
        track_id=1,
    )
    observation = TrackObservationResult(
        track_id=1,
        frame_idx=0,
        timestamp_ms=0,
        confidence=0.9,
        x_min=0.1,
        x_max=0.8,
        y_min=0.2,
        y_max=0.7,
    )
    return TrackingBatchResult(tracks=[track], observations=[observation])


def test_track_video_persists_tracking_batch() -> None:
    db = _FakeDatabase()
    tracker = _FakeTracker(_result())
    video = VideoMetadata(video_id="L21_V001", video_path=Path("video.mp4"))

    tracks = track_video(video, db, tracker)

    assert tracks == [replace(tracker.result.tracks[0], track_id=201)]
    assert tracker.calls == [(video, db.shots)]
    assert db.persisted_batches == [
        (tracker.result.tracks, tracker.result.observations)
    ]
