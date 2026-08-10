"""Unit tests for database model to pipeline contract adapters."""

import unittest
from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata, VideoMetadata
from BackEnd.app.database.adapter import (
    frame_metadata_from_frame,
    video_metadata_from_video,
)
from BackEnd.app.database.models import Frame, Video


class DatabaseAdapterTests(unittest.TestCase):
    """Verify model-to-contract conversions without a database connection."""

    def test_frame_metadata_from_frame(self) -> None:
        frame = Frame(
            frame_id="frame-1",
            video_id="video-1",
            shot_id="shot-1",
            timestamp_ms=1_000,
            fps=30.0,
            frame_idx=30,
            source="extracted",
            n=1,
            pts_time=1_000,
            frame_path="frames/frame-1.jpg",
            width=1920,
            height=1080,
        )

        result = frame_metadata_from_frame(frame)

        self.assertEqual(
            result,
            FrameMetadata(
                frame_id="frame-1",
                video_id="video-1",
                shot_id="shot-1",
                timestamp_ms=1_000,
                fps=30.0,
                frame_idx=30,
                source="extracted",
                n=1,
                pts_time=1_000,
                frame_path=Path("frames/frame-1.jpg"),
                width=1920,
                height=1080,
            ),
        )

    def test_video_metadata_from_video(self) -> None:
        video = Video(
            video_id="video-1",
            video_path="videos/video-1.mp4",
            title="Example",
            keywords=["one", "two"],
            duration_ms=10_000,
        )

        result = video_metadata_from_video(video)

        self.assertIsInstance(result, VideoMetadata)
        self.assertEqual(result.video_path, Path("videos/video-1.mp4"))
        self.assertEqual(result.keywords, ("one", "two"))
        self.assertEqual(result.title, "Example")
        self.assertEqual(result.duration_ms, 10_000)


if __name__ == "__main__":
    unittest.main()
