from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.openimages_jsonl import (
    _default_detector,
    detect_frame,
    detect_frame_to_jsonl,
    detect_frames,
)
from BackEnd.app.object_detection.schemas import BoundingBox, Detection
from BackEnd.app.object_detection.tfhub_openimages_detector import (
    TFHubOpenImagesDetector,
    _sha256_path,
)


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


class _FakeBatchDetector(_FakeDetector):
    batch_size = 4

    def __init__(self) -> None:
        self.batch_calls = 0

    def detect_batch(self, images, *, frame_ids=None, img_paths=None):
        self.batch_calls += 1
        return [
            self.detect(image, frame_id=frame_id, img_path=img_path)
            for image, frame_id, img_path in zip(images, frame_ids, img_paths)
        ]


class OpenImagesJsonlTests(unittest.TestCase):
    def test_default_detector_is_loaded_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.jpg"
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.zeros((100, 200, 3), dtype=np.uint8),
                )
            )
            frames = [
                FrameMetadata(
                    frame_id=f"frame-{index}",
                    video_id="L21_V014",
                    shot_id="L21_V014_S0013",
                    timestamp_ms=index * 1_000,
                    fps=25.0,
                    frame_idx=index * 25,
                    frame_path=image_path,
                )
                for index in range(2)
            ]

            _default_detector.cache_clear()
            with patch(
                "BackEnd.app.object_detection.openimages_jsonl."
                "TFHubOpenImagesDetector",
                return_value=_FakeDetector(),
            ) as detector_factory:
                detect_frame(frames[0])
                detect_frame(frames[1])

            _default_detector.cache_clear()
            detector_factory.assert_called_once_with()

    def test_detect_frames_uses_one_batch_and_preserves_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            frames = []
            for index in range(2):
                image_path = temporary_path / f"frame_{index}.jpg"
                self.assertTrue(
                    cv2.imwrite(
                        str(image_path),
                        np.zeros((100, 200, 3), dtype=np.uint8),
                    )
                )
                frames.append(
                    FrameMetadata(
                        frame_id=f"frame-{index}",
                        video_id="L21_V014",
                        shot_id="L21_V014_S0013",
                        timestamp_ms=index * 1_000,
                        fps=25.0,
                        frame_idx=index * 25,
                        frame_path=image_path,
                    )
                )

            detector = _FakeBatchDetector()
            contracts = detect_frames(frames, detector=detector, batch_size=2)

            self.assertEqual(detector.batch_calls, 1)
            self.assertEqual(
                [contract.frame_id for contract in contracts],
                ["frame-0", "frame-1"],
            )

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

    def test_tfhub_detector_calls_model_once_for_image_batch(self) -> None:
        received_shapes = []

        def fake_model(images):
            received_shapes.append(images.shape)
            return {
                "detection_boxes": [
                    [[0.1, 0.2, 0.5, 0.8]],
                    [[0.2, 0.1, 0.6, 0.7]],
                ],
                "detection_scores": [[[0.9]], [[0.8]]],
                "detection_class_names": [[[b"/m/01g317"]], [[b"/m/01g317"]]],
                "detection_class_entities": [[[b"Person"]], [[b"Person"]]],
                "detection_class_labels": [[[69.0]], [[69.0]]],
            }

        detector = TFHubOpenImagesDetector(
            model=fake_model,
            batch_size=2,
            supports_batching=True,
        )
        detections = detector.detect_batch(
            [
                np.zeros((100, 200, 3), dtype=np.uint8),
                np.zeros((100, 200, 3), dtype=np.uint8),
            ],
            frame_ids=["frame-1", "frame-2"],
        )

        self.assertEqual(received_shapes, [(2, 100, 200, 3)])
        self.assertEqual(
            [items[0].frame_id for items in detections],
            ["frame-1", "frame-2"],
        )

    def test_local_model_checksum_changes_with_model_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "saved_model"
            model_path.mkdir()
            weights_path = model_path / "weights.bin"
            weights_path.write_bytes(b"version-1")
            first_hash = _sha256_path(model_path)

            weights_path.write_bytes(b"version-2")
            second_hash = _sha256_path(model_path)

            self.assertEqual(len(first_hash), 64)
            self.assertNotEqual(first_hash, second_hash)
