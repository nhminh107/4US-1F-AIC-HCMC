"""Tests for the keyframe extraction demo runner helpers."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from BackEnd.app.contracts.pipeline import ShotMetadata
from scripts.run_extract_keyframe_demo import _limit_shots, _prepare_output_dir, _print_progress_event


class RunExtractKeyframeDemoTests(unittest.TestCase):
    def test_prepare_output_dir_creates_video_folder_and_checks_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "keyframes"

            video_output_dir = _prepare_output_dir(output_root, "L21_V001")

            self.assertEqual(video_output_dir, output_root / "L21_V001")
            self.assertTrue(video_output_dir.is_dir())
            self.assertFalse((video_output_dir / ".write_test.tmp").exists())

    def test_limit_shots_keeps_first_sorted_shots(self) -> None:
        shots = [
            ShotMetadata("V001_S003", "V001", 3, 3000, 4000, 30, 39),
            ShotMetadata("V001_S001", "V001", 1, 1000, 2000, 10, 19),
            ShotMetadata("V001_S002", "V001", 2, 2000, 3000, 20, 29),
        ]

        limited = _limit_shots(shots, 2)

        self.assertEqual([shot.shot_id for shot in limited], ["V001_S001", "V001_S002"])

    def test_limit_shots_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            _limit_shots([], 0)

    def test_print_progress_event_reports_video_level_export(self) -> None:
        event = {
            "phase": "export",
            "frame_count": 307,
            "chunk_count": 4,
            "chunk_size": 100,
            "export_s": 12.345,
        }
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            _print_progress_event(event)

        text = output.getvalue()
        self.assertIn("Final FFmpeg export", text)
        self.assertIn("frames=307", text)
        self.assertIn("chunks=4", text)
        self.assertIn("export=12.35s", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
