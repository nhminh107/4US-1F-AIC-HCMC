"""Tests for the lightweight clip extraction pipeline."""

from __future__ import annotations

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.pipeline.extract_clip import extract_clip


class _FakeDatabase:
    def __init__(self) -> None:
        self.saved_clips: list[dict[str, object]] = []

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        return [
            ShotMetadata(
                shot_id=f"{video_id}_S000",
                video_id=video_id,
                shot_index=0,
                start_ms=0,
                end_ms=2_000,
                start_frame_idx=0,
                end_frame_idx=59,
            )
        ]

    def add_clip(self, **kwargs: object) -> None:
        self.saved_clips.append(kwargs)


class _FakeClipExtractor:
    def run(self, shot: ShotMetadata) -> list[dict[str, object]]:
        return [
            {
                "clip_id": "L21V005S000C01",
                "shot_id": shot.shot_id,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
                "start_frame_idx": shot.start_frame_idx,
                "end_frame_idx": shot.end_frame_idx,
                "sampling_fps": 2.0,
                "clip_path": "data/clips/L21_V005/L21V005S000C01.mp4",
            }
        ]


def test_extract_clip_persists_and_returns_pipeline_contracts() -> None:
    database = _FakeDatabase()

    clips = extract_clip("L21_V005", database, _FakeClipExtractor())  # type: ignore[arg-type]

    assert len(clips) == 1
    assert clips[0].clip_id == "L21V005S000C01"
    assert str(clips[0].clip_path) == "data/clips/L21_V005/L21V005S000C01.mp4"
    assert database.saved_clips == [
        {
            "clip_id": "L21V005S000C01",
            "shot_id": "L21_V005_S000",
            "start_ms": 0,
            "end_ms": 2_000,
            "start_frame_idx": 0,
            "end_frame_idx": 59,
            "sampling_fps": 2.0,
            "clip_path": "data/clips/L21_V005/L21V005S000C01.mp4",
        }
    ]
