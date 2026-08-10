from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata, VideoMetadata
from BackEnd.app.object_detection.schemas import BoundingBox, Detection
from BackEnd.app.tracking.CONFIG import TrackingConfig
from BackEnd.app.tracking.tracking import ByteTrackService


class _FakeDetector:
    model_name = "fake"
    model_version = "test"

    def detect(self, image, *, frame_id=None, img_path=None):
        return [
            Detection(
                bbox=BoundingBox(10, 10, 40, 40),
                confidence=0.9,
                class_index=69,
                class_id="/m/01g317",
                class_name="Person",
                frame_id=frame_id,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        ]


class ByteTrackServiceTests(unittest.TestCase):
    def test_decodes_once_and_resets_tracker_for_each_shot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.touch()
            video = VideoMetadata(video_id="L21_V001", video_path=video_path)
            shots = [
                ShotMetadata("shot-1", video.video_id, 0, 0, 1_500),
                ShotMetadata("shot-2", video.video_id, 1, 1_500, 3_000),
            ]
            frames = [
                (timestamp_ms, index, np.zeros((60, 80, 3), dtype=np.uint8))
                for index, timestamp_ms in enumerate(range(0, 3_000, 500))
            ]
            service = ByteTrackService(
                detector=_FakeDetector(),
                config=TrackingConfig(sampling_fps=2.0),
            )

            with patch(
                "BackEnd.app.tracking.tracking._iter_video_frames",
                return_value=iter(frames),
            ) as decode:
                result = service.track_video(video, shots)

            decode.assert_called_once_with(video_path)
            self.assertEqual(len(result.detections), 6)
            self.assertEqual(len(result.tracks), 2)
            self.assertEqual(
                [track.observation_count for track in result.tracks],
                [2, 2],
            )
            self.assertEqual(
                {track.shot_id for track in result.tracks},
                {"shot-1", "shot-2"},
            )
            self.assertEqual(len(result.observations), 4)

    def test_rejects_overlapping_shots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.touch()
            video = VideoMetadata(video_id="L21_V001", video_path=video_path)
            shots = [
                ShotMetadata("shot-1", video.video_id, 0, 0, 2_000),
                ShotMetadata("shot-2", video.video_id, 1, 1_000, 3_000),
            ]
            service = ByteTrackService(detector=_FakeDetector())

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                service.track_video(video, shots)

    def test_uses_persisted_frames_as_tracking_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.touch()
            video = VideoMetadata(video_id="L21_V001", video_path=video_path)
            shots = [ShotMetadata("shot-1", video.video_id, 0, 0, 1_000)]
            persisted_frame = FrameMetadata(
                frame_id="L21_V001_E001",
                video_id=video.video_id,
                shot_id="shot-1",
                timestamp_ms=250,
                fps=25.0,
                frame_idx=6,
                source="extracted",
            )
            decoded_frames = [
                (timestamp_ms, index, np.zeros((60, 80, 3), dtype=np.uint8))
                for index, timestamp_ms in enumerate((0, 500))
            ]
            persisted_event = (
                250,
                6,
                np.zeros((60, 80, 3), dtype=np.uint8),
                persisted_frame.frame_id,
            )
            service = ByteTrackService(
                detector=_FakeDetector(),
                config=TrackingConfig(sampling_fps=2.0),
            )

            with (
                patch(
                    "BackEnd.app.tracking.tracking._iter_video_frames",
                    return_value=iter(decoded_frames),
                ),
                patch(
                    "BackEnd.app.tracking.tracking._load_persisted_frame_events",
                    return_value=[persisted_event],
                ) as load_events,
            ):
                result = service.track_video(video, shots, [persisted_frame])

            load_events.assert_called_once_with(video.video_id, [persisted_frame])
            self.assertEqual(len(result.detections), 3)
            self.assertIn(
                persisted_frame.frame_id,
                {detection.frame_id for detection in result.detections},
            )
