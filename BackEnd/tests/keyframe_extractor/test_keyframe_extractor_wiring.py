"""Wiring tests bổ sung cho KeyframeExtractor với mock decode FFmpeg."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor


class KeyframeExtractorWiringTests(unittest.TestCase):
    """Kiểm tra orchestrator của KeyframeExtractor với nhiều kịch bản testcase."""

    def test_extract_returns_valid_frame_metadata(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")

        shot = ShotMetadata(
            shot_id="L21_V001_S001",
            video_id="L21_V001",
            shot_index=1,
            start_ms=0,
            end_ms=10000,
            start_frame_idx=0,
            end_frame_idx=249,
        )

        fake_dimensions = [(1920, 1080)] * 5

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: fake_dimensions[: len(indices)],
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        for frame in frames:
            self.assertEqual(frame.video_id, "L21_V001")
            self.assertEqual(frame.shot_id, "L21_V001_S001")
            self.assertEqual(frame.frame_role, "keyframe")
            self.assertEqual(frame.source, "extracted")
            self.assertEqual(frame.width, 1920)
            self.assertEqual(frame.height, 1080)
            self.assertTrue(frame.frame_id.startswith("L21_V001_E"))
            self.assertLessEqual(len(frame.frame_id), 15, "frame_id phải vừa varchar(15)")

    def test_extract_for_video_sequences_seq_numbers(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")

        shots = [
            ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124),
            ShotMetadata("L21_V001_S002", "L21_V001", 1, 5000, 10000, 125, 249),
        ]

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract_for_video("L21_V001", shots, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        seqs = [f.frame_id.split("_E")[1] for f in frames]
        self.assertEqual(seqs, [f"{i:03d}" for i in range(1, len(frames) + 1)])

    def test_extract_for_video_empty_shots_returns_empty(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        with patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0):
            frames = extractor.extract_for_video("L21_V001", [], existing_frame_idxs=[])
        self.assertEqual(frames, [])

    def test_extract_raises_error_if_start_frame_idx_is_none(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        shot_invalid = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, None, 124)
        with self.assertRaises(ValueError):
            extractor.extract(shot_invalid)

    def test_custom_seq_start_parameter(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        shot = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)
        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[], seq_start=10)
        self.assertTrue(frames[0].frame_id.endswith("_E010"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
