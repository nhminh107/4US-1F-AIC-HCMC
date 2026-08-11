"""Unit tests for independent YOLO26 tracking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from BackEnd.app.contracts.pipeline import ShotMetadata, VideoMetadata
from BackEnd.app.tracking.CONFIG import TrackingConfig
from BackEnd.app.tracking.class_mapping import COCO_TO_OPENIMAGES
from BackEnd.app.tracking.tracking import YOLOTrackingService


class _FakeUltralyticsTracker:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _FakeBoxes:
    def __init__(
        self,
        *,
        track_id: int = 7,
        class_index: int = 0,
        confidence: float = 0.9,
    ) -> None:
        self.is_track = True
        self.id = np.asarray([track_id], dtype=np.float32)
        self.cls = np.asarray([class_index], dtype=np.float32)
        self.conf = np.asarray([confidence], dtype=np.float32)
        self.xyxy = np.asarray([[10.0, 10.0, 40.0, 40.0]], dtype=np.float32)


class _FakeYOLOModel:
    def __init__(self) -> None:
        self.names = {
            index: mapped_class.coco_name
            for index, mapped_class in COCO_TO_OPENIMAGES.items()
        }
        self.predictor = None
        self.calls: list[dict[str, object]] = []
        self.tracker = _FakeUltralyticsTracker()

    def track(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        if self.predictor is None:
            self.predictor = SimpleNamespace(trackers=[self.tracker])
        return [SimpleNamespace(boxes=_FakeBoxes())]


class YOLOTrackingServiceTests(unittest.TestCase):
    def _service(self, model: _FakeYOLOModel, sampling_fps: float = 2.0):
        return YOLOTrackingService(
            model=model,
            config=TrackingConfig(
                model_path=Path("fake-yolo26n.pt"),
                sampling_fps=sampling_fps,
            ),
        )

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
            model = _FakeYOLOModel()
            service = self._service(model)

            with patch(
                "BackEnd.app.tracking.tracking._iter_video_frames",
                return_value=iter(frames),
            ) as decode:
                result = service.track_video(video, shots)

            decode.assert_called_once_with(video_path)
            self.assertEqual(len(model.calls), 6)
            self.assertEqual(len(result.tracks), 2)
            self.assertEqual(
                [track.observation_count for track in result.tracks],
                [3, 3],
            )
            self.assertEqual(len(result.observations), 6)
            self.assertEqual(model.tracker.reset_count, 1)
            self.assertEqual(
                {track.shot_id for track in result.tracks},
                {"shot-1", "shot-2"},
            )
            self.assertTrue(
                all(track.class_id == "/m/01g317" for track in result.tracks)
            )
            self.assertEqual(result.observations[0].x_min, 0.125)
            self.assertEqual(result.observations[0].y_min, 1 / 6)
            self.assertEqual(result.observations[0].x_max, 0.5)
            self.assertEqual(result.observations[0].y_max, 2 / 3)
            self.assertTrue(
                all(call["persist"] is True for call in model.calls)
            )
            self.assertTrue(
                all(
                    call["classes"] == list(service.config.class_indices)
                    for call in model.calls
                )
            )
            self.assertEqual(len(service.config.class_indices), 22)
            self.assertTrue(
                all(
                    str(call["tracker"]).endswith("tracking/bytetrack.yaml")
                    for call in model.calls
                )
            )

    def test_sampling_controls_the_number_of_yolo_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.touch()
            video = VideoMetadata(video_id="L21_V001", video_path=video_path)
            shots = [ShotMetadata("shot-1", video.video_id, 0, 0, 1_000)]
            frames = [
                (timestamp_ms, index, np.zeros((60, 80, 3), dtype=np.uint8))
                for index, timestamp_ms in enumerate((0, 100, 499, 500, 900))
            ]
            model = _FakeYOLOModel()
            service = self._service(model, sampling_fps=2.0)

            with patch(
                "BackEnd.app.tracking.tracking._iter_video_frames",
                return_value=iter(frames),
            ):
                result = service.track_video(video, shots)

            self.assertEqual(len(model.calls), 2)
            self.assertEqual(
                [item.timestamp_ms for item in result.observations],
                [0, 500],
            )

    def test_rejects_overlapping_shots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.touch()
            video = VideoMetadata(video_id="L21_V001", video_path=video_path)
            shots = [
                ShotMetadata("shot-1", video.video_id, 0, 0, 2_000),
                ShotMetadata("shot-2", video.video_id, 1, 1_000, 3_000),
            ]
            service = self._service(_FakeYOLOModel())

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                service.track_video(video, shots)

    def test_requires_local_weight_when_model_is_not_injected(self) -> None:
        missing_weight = Path("definitely-missing-yolo26n.pt")

        with self.assertRaisesRegex(FileNotFoundError, "automatic model download"):
            YOLOTrackingService(config=TrackingConfig(model_path=missing_weight))


if __name__ == "__main__":
    unittest.main()
