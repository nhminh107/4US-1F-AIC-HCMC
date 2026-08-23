"""Tests for the database adapter around additional-keyframe extraction."""

from __future__ import annotations

import unittest
from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata
from BackEnd.app.pipeline.extract_keyframe import extract_keyframes


class _FakeDatabase:
    def __init__(self, existing: list[FrameMetadata]) -> None:
        self.existing = existing
        self.added: list[dict[str, object]] = []

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        return self.existing

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        return [ShotMetadata("V001_S001", video_id, 0, 0, 1000, 0, 24)]

    def add_frame(self, **kwargs: object) -> None:
        self.added.append(kwargs)


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.last_selection_manifest = [{"shot_id": "V001_S001", "selected_frame_idxs": [20]}]

    def extract_for_video(self, **kwargs: object) -> list[FrameMetadata]:
        self.calls.append(kwargs)
        return [
            FrameMetadata(
                frame_id="V001_E003",
                video_id="V001",
                shot_id="V001_S001",
                timestamp_ms=800,
                fps=25.0,
                frame_idx=20,
                source="extracted",
                n=20,
                pts_time=0.8,
                frame_path=Path("data/keyframes/V001/V001_E003.jpg"),
                width=1920,
                height=1080,
            )
        ]


class PipelineKeyframeAdapterTests(unittest.TestCase):
    def test_passes_existing_organizer_and_extracted_frames_without_new_pipeline_output(self) -> None:
        existing = [
            FrameMetadata("V001_001", "V001", "V001_S001", 0, 25.0, 0, "official", 0, 0.0, None, 1920, 1080),
            FrameMetadata("V001_E002", "V001", "V001_S001", 400, 25.0, 10, "extracted", 10, 0.4, None, 1920, 1080),
        ]
        database = _FakeDatabase(existing)
        extractor = _FakeExtractor()

        extract_keyframes("V001", database, extractor)  # type: ignore[arg-type]

        self.assertEqual(extractor.calls[0]["existing_frame_idxs"], [0, 10])
        self.assertEqual(extractor.calls[0]["existing_frame_ids"], ["V001_001", "V001_E002"])
        self.assertEqual(database.added[0]["frame_id"], "V001_E003")


if __name__ == "__main__":
    unittest.main()
