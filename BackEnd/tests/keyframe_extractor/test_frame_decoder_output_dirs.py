"""Regression tests for keyframe output directory creation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from BackEnd.app.keyframe_extractor.frame_decoder import (
    _extract_single_frame,
    extract_and_save_frames,
    extract_and_save_frames_chunked,
)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg not found on PATH")
class FrameDecoderOutputDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.video_path = self.temp_dir / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=5:duration=1",
                "-pix_fmt",
                "yuv420p",
                str(self.video_path),
            ],
            capture_output=True,
            check=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_and_save_frames_creates_missing_parent_directory(self) -> None:
        output_path = self.temp_dir / "missing" / "nested" / "frame.jpg"

        dimensions = extract_and_save_frames(self.video_path, [0], [output_path])

        self.assertEqual(dimensions, [(64, 48)])
        self.assertTrue(output_path.is_file())

    def test_single_frame_helper_creates_missing_parent_directory(self) -> None:
        output_path = self.temp_dir / "single" / "nested" / "frame.jpg"

        _extract_single_frame(self.video_path, 0, output_path)

        self.assertTrue(output_path.is_file())


class FrameDecoderChunkedExportTests(unittest.TestCase):
    def test_chunked_export_preserves_order_and_splits_batches(self) -> None:
        video_path = Path("sample.mp4")
        frame_indices = [10, 20, 30, 40, 50]
        output_paths = [Path(f"frame_{idx}.jpg") for idx in frame_indices]

        def fake_extract(video, indices, paths):
            return [(idx, idx + 1) for idx in indices]

        with patch(
            "BackEnd.app.keyframe_extractor.frame_decoder.extract_and_save_frames",
            side_effect=fake_extract,
        ) as extract_mock:
            dimensions = extract_and_save_frames_chunked(
                video_path,
                frame_indices,
                output_paths,
                chunk_size=2,
            )

        self.assertEqual(dimensions, [(10, 11), (20, 21), (30, 31), (40, 41), (50, 51)])
        self.assertEqual(extract_mock.call_count, 3)
        self.assertEqual(list(extract_mock.call_args_list[0].args[1]), [10, 20])
        self.assertEqual(list(extract_mock.call_args_list[1].args[1]), [30, 40])
        self.assertEqual(list(extract_mock.call_args_list[2].args[1]), [50])

    def test_chunked_export_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            extract_and_save_frames_chunked(
                Path("sample.mp4"),
                [1],
                [Path("frame.jpg")],
                chunk_size=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
