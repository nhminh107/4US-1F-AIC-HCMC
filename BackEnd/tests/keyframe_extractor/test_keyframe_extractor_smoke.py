"""Smoke test end-to-end cho KeyframeExtractor với video local thật và FFmpeg thật."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.keyframe_extractor import DEFAULT_VIDEO_DIR, KeyframeExtractor

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _first_sample_video_id() -> str | None:
    if not DEFAULT_VIDEO_DIR.is_dir():
        return None
    videos = sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    return videos[0].stem if videos else None


_SAMPLE_VIDEO_ID = _first_sample_video_id()


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg/ffprobe not found on PATH")
@unittest.skipUnless(_SAMPLE_VIDEO_ID is not None, f"no sample .mp4 under {DEFAULT_VIDEO_DIR}")
class KeyframeExtractorSmokeTests(unittest.TestCase):
    """Chạy KeyframeExtractor end-to-end trên video thật và FFmpeg thật."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.keyframe_dir = Path(self.temp_dir) / "keyframes"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_creates_real_jpeg_files(self) -> None:
        extractor = KeyframeExtractor(keyframe_dir=self.keyframe_dir)

        # Shot giả định từ 0 đến 124 (5 giây đầu của video mẫu)
        shot = ShotMetadata(
            shot_id=f"{_SAMPLE_VIDEO_ID}_S001",
            video_id=_SAMPLE_VIDEO_ID,
            shot_index=1,
            start_ms=0,
            end_ms=5000,
            start_frame_idx=0,
            end_frame_idx=124,
        )

        frames = extractor.extract(shot, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        for frame in frames:
            self.assertEqual(frame.video_id, _SAMPLE_VIDEO_ID)
            self.assertEqual(frame.source, "extracted")
            self.assertEqual(frame.frame_role, "keyframe")
            self.assertIsNotNone(frame.frame_path)
            self.assertTrue(frame.frame_path.is_file(), f"Image file missing: {frame.frame_path}")
            self.assertGreater(frame.width, 0)
            self.assertGreater(frame.height, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
