from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import VideoAsset
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder
from BackEnd.app.embedding.clip.video_repository import VideoRepository
from BackEnd.app.embedding.clip.decoder import _group_timestamps
from BackEnd.app.embedding.common.errors import MediaNotFoundError


class VideoRepositoryTests(unittest.TestCase):
    def test_resolve_video_finds_flat_mp4_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "L21_V001.mp4"
            video_path.write_bytes(b"not a real video")

            asset = VideoRepository(temp_dir).resolve_video("L21_V001")

            self.assertEqual(asset.video_id, "L21_V001")
            self.assertEqual(asset.video_uri, video_path)

    def test_resolve_video_reports_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(MediaNotFoundError):
                VideoRepository(temp_dir).resolve_video("L21_V001")

    def test_decoder_timestamp_grouping_reduces_close_seeks(self) -> None:
        self.assertEqual(
            _group_timestamps([100, 500, 3000, 3300], max_gap_ms=1000),
            [[100, 500], [3000, 3300]],
        )

    def test_pyav_decoder_reads_synthetic_video(self) -> None:
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is not installed.")

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "synthetic.mp4"
            _write_synthetic_video(video_path, av)

            batch = PyAVVideoDecoder().decode_nearest_frames(
                VideoAsset(video_id="synthetic", video_uri=video_path),
                (100, 300, 500),
            )

            self.assertEqual(batch.requested_timestamps_ms, (100, 300, 500))
            self.assertTrue(any(status == "success" for status in batch.decode_statuses))
            self.assertGreaterEqual(batch.metrics["seek_count"], 1)


def _write_synthetic_video(video_path: Path, av_module) -> None:
    with av_module.open(str(video_path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            image = np.full((64, 64, 3), index * 20, dtype=np.uint8)
            frame = av_module.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
