"""Tests for the thin object-detection persistence pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from BackEnd.app.contracts.pipeline import (
    FrameMetadata,
    ObjectDetectionResult,
)
from BackEnd.app.pipeline.object_detection import detect_object, detect_objects


class _FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def add_object_detection(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(detection_id=len(self.calls))


def _frame(image_path: Path, frame_id: str = "L21_V001_E001") -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V001",
        shot_id="L21_V001_S000",
        timestamp_ms=0,
        fps=25.0,
        frame_idx=0,
        source="extracted",
        frame_path=image_path,
    )


def _detection(frame_id: str) -> ObjectDetectionResult:
    return ObjectDetectionResult(
        frame_id=frame_id,
        class_id="/m/01g317",
        confidence=0.9,
        x_min=0.1,
        x_max=0.8,
        y_min=0.2,
        y_max=0.7,
        model_name="fake-openimages",
        model_version="v1",
    )


class ObjectDetectionPipelineTests(unittest.TestCase):
    def test_detect_object_calls_database_insert(self) -> None:
        db = _FakeDatabase()
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.jpg"
            image_path.touch()
            frame = _frame(image_path)
            expected = _detection(frame.frame_id)
            with patch(
                "BackEnd.app.pipeline.object_detection.detect_frame",
                return_value=[expected],
            ) as mocked_detect:
                results = detect_object(frame, db, detector=object())

        self.assertEqual(results, [replace(expected, detection_id=1)])
        self.assertEqual(db.calls[0]["frame_id"], expected.frame_id)
        self.assertEqual(db.calls[0]["class_id"], expected.class_id)
        mocked_detect.assert_called_once()

    def test_detect_objects_returns_flattened_results(self) -> None:
        db = _FakeDatabase()
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.jpg"
            image_path.touch()
            frames = [_frame(image_path), _frame(image_path, "L21_V001_E002")]
            returned = [_detection(frame.frame_id) for frame in frames]
            with patch(
                "BackEnd.app.pipeline.object_detection.detect_frame",
                side_effect=[[returned[0]], [returned[1]]],
            ):
                results = detect_objects(frames, db, detector=object())

        self.assertEqual(
            results,
            [replace(returned[0], detection_id=1), replace(returned[1], detection_id=2)],
        )
        self.assertEqual(len(db.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
