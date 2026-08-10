from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.openimages_jsonl import detect_frame, detect_frame_to_jsonl
from BackEnd.app.object_detection.schemas import BoundingBox, Detection
from BackEnd.app.object_detection.tfhub_openimages_detector import TFHubOpenImagesDetector


class _FakeDetector:
    model_name = "fake"
    model_version = "test"

    def detect(self, image: np.ndarray, *, frame_id: str | None = None, img_path: str | None = None) -> list[Detection]:
        return [
            Detection(
                bbox=BoundingBox(x_min=20, y_min=10, x_max=100, y_max=50),
                confidence=0.9,
                class_index=69,
                class_id="/m/01g317",
                class_name="Person",
                frame_id=frame_id,
                img_path=img_path,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        ]


class OpenImagesJsonlTests(unittest.TestCase):
    def test_detect_frame_returns_contracts_without_writing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_001.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8)))
            frame = FrameMetadata(
                frame_id="L21_V014_013",
                video_id="L21_V014",
                shot_id="L21_V014_S0013",
                timestamp_ms=0,
                fps=25.0,
                frame_idx=0,
                frame_path=image_path,
            )

            contracts = detect_frame(frame, detector=_FakeDetector())

            self.assertEqual(len(contracts), 1)
            self.assertIsInstance(contracts[0], ObjectDetectionResult)
            self.assertEqual(contracts[0].frame_id, frame.frame_id)
            self.assertIsNone(contracts[0].detection_id)
            self.assertEqual(list(temporary_path.glob("*.jsonl")), [])

    def test_one_image_writes_object_detection_contract_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_001.jpg"
            output_path = temporary_path / "result.jsonl"
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            frame = FrameMetadata(
                frame_id="L21_V014_013",
                video_id="L21_V014",
                shot_id="L21_V014_S0013",
                timestamp_ms=0,
                fps=25.0,
                frame_idx=0,
                frame_path=image_path,
                width=200,
                height=100,
            )
            contracts = detect_frame_to_jsonl(
                frame,
                output_path,
                detector=_FakeDetector(),
            )

            self.assertEqual(len(contracts), 1)
            self.assertIsInstance(contracts[0], ObjectDetectionResult)
            self.assertEqual(contracts[0].class_id, "/m/01g317")
            self.assertEqual(contracts[0].x_min, 0.1)
            self.assertEqual(contracts[0].x_max, 0.5)
            self.assertEqual(contracts[0].y_min, 0.1)
            self.assertEqual(contracts[0].y_max, 0.5)

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records, [
                {
                    "frame_id": "L21_V014_013",
                    "class_id": "/m/01g317",
                    "confidence": 0.9,
                    "x_min": 0.1,
                    "x_max": 0.5,
                    "y_min": 0.1,
                    "y_max": 0.5,
                    "model_name": "fake",
                    "model_version": "test",
                    "detection_id": 1,
                }
            ])

    def test_tfhub_output_uses_openimages_mid_and_entity_name(self) -> None:
        detector = TFHubOpenImagesDetector(
            confidence_threshold=0.25,
            model=lambda image: {
                "detection_boxes": [[0.1, 0.2, 0.5, 0.8]],
                "detection_scores": [[0.9]],
                "detection_class_names": [[b"/m/01g317"]],
                "detection_class_entities": [[b"Person"]],
                "detection_class_labels": [[69.0]],
            },
        )

        detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), frame_id="frame-1")

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, "/m/01g317")
        self.assertEqual(detections[0].class_name, "Person")
        self.assertEqual(detections[0].bbox.to_xyxy(), [40.0, 10.0, 160.0, 50.0])
