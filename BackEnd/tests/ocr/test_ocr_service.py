"""Contract, geometry, and batching tests for the OCR service."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import cv2
import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.ocr.config import OCRConfig
from BackEnd.app.ocr.engine import OCREngine, PaddleOCREngine, VietOCRTextRecognizer
from BackEnd.app.ocr.geometry import normalized_bounding_box, perspective_crop
from BackEnd.app.ocr.run_ocr_pipeline import run_ocr, run_ocr_batch
from BackEnd.app.ocr.schemas import DetectedTextRegion, RecognizedText
from BackEnd.app.ocr.service import OCRService, normalize_text


class FakeOCREngine(OCREngine):
    """Deterministic in-memory engine used to verify orchestration."""

    def __init__(
        self,
        detections: list[list[DetectedTextRegion]],
        recognitions: list[RecognizedText],
    ) -> None:
        self.detections = detections
        self.recognitions = recognitions
        self.detect_calls = 0
        self.recognize_calls = 0
        self.detect_batch_size: int | None = None
        self.recognize_batch_size: int | None = None
        self.recognized_crop_count = 0

    @property
    def model_version(self) -> str:
        return "fake-1.0"

    def detect(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[list[DetectedTextRegion]]:
        self.detect_calls += 1
        self.detect_batch_size = batch_size
        if len(images) != len(self.detections):
            raise AssertionError("Unexpected detection input count.")
        return self.detections

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        self.recognize_calls += 1
        self.recognize_batch_size = batch_size
        self.recognized_crop_count = len(images)
        if len(images) != len(self.recognitions):
            raise AssertionError("Unexpected recognition input count.")
        return self.recognitions


def make_region(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float = 0.9,
) -> DetectedTextRegion:
    return DetectedTextRegion(
        polygon=np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        ),
        confidence=confidence,
    )


def make_frame(frame_id: str, frame_path: Path, *, width: int = 200) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V001",
        shot_id="L21_V001_S001",
        timestamp_ms=1000,
        fps=25.0,
        frame_idx=25,
        source="official",
        n=1,
        frame_path=frame_path,
        width=width,
        height=100,
    )


class OCRServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        self.first_path = directory / "first.jpg"
        self.second_path = directory / "second.jpg"
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 10), (100, 40), (0, 0, 0), thickness=-1)
        self.assertTrue(cv2.imwrite(str(self.first_path), image))
        self.assertTrue(cv2.imwrite(str(self.second_path), image))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_single_frame_accepts_and_returns_pipeline_contracts(self) -> None:
        engine = FakeOCREngine(
            detections=[[make_region(20, 10, 100, 40)]],
            recognitions=[RecognizedText("  Thành   phố Hồ Chí Minh  ", 0.95)],
        )
        service = OCRService(engine=engine)

        results = service.process_frame(make_frame("frame_001", self.first_path))

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], OCRResult)
        self.assertEqual(results[0].frame_id, "frame_001")
        self.assertEqual(results[0].n, 0)
        self.assertEqual(results[0].text, "Thành phố Hồ Chí Minh")
        self.assertEqual(results[0].language, "vi")
        self.assertAlmostEqual(results[0].x_min, 0.1)
        self.assertAlmostEqual(results[0].x_max, 0.5)
        self.assertAlmostEqual(results[0].y_min, 0.1)
        self.assertAlmostEqual(results[0].y_max, 0.4)

    def test_batch_detects_once_and_recognizes_all_crops_once(self) -> None:
        engine = FakeOCREngine(
            detections=[
                [make_region(10, 60, 80, 90), make_region(20, 10, 100, 40)],
                [make_region(30, 20, 150, 50)],
            ],
            recognitions=[
                RecognizedText("headline", 0.9),
                RecognizedText("ticker", 0.8),
                RecognizedText("second frame", 0.85),
            ],
        )
        config = OCRConfig(detection_batch_size=2, recognition_batch_size=16)
        service = OCRService(config, engine=engine)
        frames = [
            make_frame("frame_001", self.first_path),
            make_frame("frame_002", self.second_path),
        ]

        results = service.process_batch(frames)

        self.assertEqual(engine.detect_calls, 1)
        self.assertEqual(engine.recognize_calls, 1)
        self.assertEqual(engine.detect_batch_size, 2)
        self.assertEqual(engine.recognize_batch_size, 16)
        self.assertEqual(engine.recognized_crop_count, 3)
        self.assertEqual(
            [(result.frame_id, result.n, result.text) for result in results],
            [
                ("frame_001", 0, "headline"),
                ("frame_001", 1, "ticker"),
                ("frame_002", 0, "second frame"),
            ],
        )

    def test_low_detection_and_recognition_scores_are_filtered(self) -> None:
        engine = FakeOCREngine(
            detections=[
                [
                    make_region(10, 10, 60, 30, confidence=0.2),
                    make_region(70, 10, 120, 30, confidence=0.9),
                    make_region(10, 50, 80, 80, confidence=0.9),
                ]
            ],
            recognitions=[
                RecognizedText("too uncertain", 0.1),
                RecognizedText("accepted", 0.9),
            ],
        )
        service = OCRService(engine=engine)

        results = service.process_frame(make_frame("frame_001", self.first_path))

        self.assertEqual([(result.n, result.text) for result in results], [(0, "accepted")])
        self.assertEqual(engine.recognized_crop_count, 2)

    def test_empty_batch_does_not_initialize_or_call_engine(self) -> None:
        service = OCRService()
        self.assertEqual(service.process_batch([]), [])
        self.assertIsNone(service._engine)

    def test_missing_frame_path_is_reported_with_frame_identity(self) -> None:
        frame = make_frame("missing_frame", self.first_path)
        frame = FrameMetadata(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            shot_id=frame.shot_id,
            timestamp_ms=frame.timestamp_ms,
            fps=frame.fps,
            frame_idx=frame.frame_idx,
        )
        service = OCRService(engine=FakeOCREngine([], []))

        with self.assertRaisesRegex(ValueError, "missing_frame"):
            service.process_frame(frame)

    def test_image_dimension_mismatch_is_rejected(self) -> None:
        service = OCRService(engine=FakeOCREngine([], []))
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            service.process_frame(make_frame("frame_001", self.first_path, width=201))

    def test_duplicate_frame_ids_are_rejected_before_inference(self) -> None:
        service = OCRService(engine=FakeOCREngine([], []))
        frames = [
            make_frame("frame_001", self.first_path),
            make_frame("frame_001", self.second_path),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
            service.process_batch(frames)

    def test_public_facades_preserve_contract_types(self) -> None:
        single_engine = FakeOCREngine(
            [[make_region(20, 10, 100, 40)]],
            [RecognizedText("single", 0.9)],
        )
        single_service = OCRService(engine=single_engine)
        single_results = run_ocr(
            make_frame("frame_001", self.first_path),
            service=single_service,
        )

        batch_engine = FakeOCREngine(
            [[make_region(20, 10, 100, 40)]],
            [RecognizedText("batch", 0.9)],
        )
        batch_service = OCRService(engine=batch_engine)
        batch_results = run_ocr_batch(
            [make_frame("frame_002", self.second_path)],
            service=batch_service,
        )

        self.assertTrue(all(isinstance(item, OCRResult) for item in single_results))
        self.assertTrue(all(isinstance(item, OCRResult) for item in batch_results))


class OCRGeometryAndContractTests(unittest.TestCase):
    def test_perspective_crop_rectifies_rotated_polygon(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        polygon = np.array([[20, 20], [120, 10], [125, 50], [25, 60]], dtype=np.float32)

        crop = perspective_crop(image, polygon, padding_ratio=0.04)

        self.assertEqual(crop.ndim, 3)
        self.assertGreater(crop.shape[1], crop.shape[0])
        self.assertGreaterEqual(crop.shape[0], 3)

    def test_out_of_bounds_polygon_is_clipped_before_normalization(self) -> None:
        polygon = np.array([[-10, -5], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        x_min, x_max, y_min, y_max = normalized_bounding_box(polygon, 200, 100)
        self.assertEqual(x_min, 0.0)
        self.assertEqual(y_min, 0.0)
        self.assertEqual(x_max, 0.5)
        self.assertEqual(y_max, 0.5)

    def test_unicode_normalization_preserves_vietnamese_accents(self) -> None:
        decomposed = "Tha\u0300nh   pho\u0302\u0301"
        self.assertEqual(normalize_text(decomposed), "Thành phố")

    def test_ocr_result_is_frozen_and_validates_coordinates(self) -> None:
        result = OCRResult("frame_001", 0, "text", 0.1, 0.5, 0.2, 0.6, "vi")
        with self.assertRaises(FrozenInstanceError):
            result.text = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "positive area"):
            OCRResult("frame_001", 0, "text", 0.5, 0.5, 0.2, 0.6, "vi")

    def test_config_rejects_invalid_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            OCRConfig(recognition_batch_size=0)


class FakePaddlePredictor:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.last_batch_size: int | None = None
        self.closed = False

    def predict(
        self,
        images: list[np.ndarray],
        *,
        batch_size: int,
    ) -> list[dict[str, object]]:
        self.last_batch_size = batch_size
        if len(images) != len(self.results):
            raise AssertionError("Unexpected Paddle predictor input count.")
        return self.results

    def close(self) -> None:
        self.closed = True


class FakeRecognitionBackend:
    def __init__(self, results: list[RecognizedText]) -> None:
        self.results = results
        self.last_batch_size: int | None = None
        self.closed = False

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        self.last_batch_size = batch_size
        if len(images) != len(self.results):
            raise AssertionError("Unexpected recognition backend input count.")
        return self.results

    def close(self) -> None:
        self.closed = True


class PaddleOCREngineAdapterTests(unittest.TestCase):
    def test_adapter_parses_paddle_results_and_forwards_batch_sizes(self) -> None:
        detector = FakePaddlePredictor(
            [
                {
                    "dt_polys": [
                        np.array([[1, 2], [20, 2], [20, 10], [1, 10]])
                    ],
                    "dt_scores": [0.91],
                }
            ]
        )
        recognizer = FakeRecognitionBackend([RecognizedText("Việt Nam", 0.87)])
        engine = object.__new__(PaddleOCREngine)
        engine.config = OCRConfig()
        engine._detector = detector
        engine._recognizer = recognizer
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        detections = engine.detect([image], batch_size=3)
        recognitions = engine.recognize([image], batch_size=12)

        self.assertEqual(detector.last_batch_size, 3)
        self.assertEqual(recognizer.last_batch_size, 12)
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(detections[0]), 1)
        self.assertAlmostEqual(detections[0][0].confidence, 0.91)
        self.assertEqual(recognitions, [RecognizedText("Việt Nam", 0.87)])

        engine.close()
        self.assertTrue(detector.closed)
        self.assertTrue(recognizer.closed)

    def test_adapter_rejects_mismatched_detection_fields(self) -> None:
        detector = FakePaddlePredictor(
            [{"dt_polys": [np.zeros((4, 2))], "dt_scores": []}]
        )
        engine = object.__new__(PaddleOCREngine)
        engine.config = OCRConfig()
        engine._detector = detector
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "mismatched polygons"):
            engine.detect([image], batch_size=1)


class FakeVietOCRPredictor:
    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []

    def predict_batch(
        self,
        images: list[object],
        *,
        return_prob: bool,
    ) -> tuple[list[str], list[float]]:
        self.assert_return_prob(return_prob)
        self.chunk_sizes.append(len(images))
        return ["text"] * len(images), [0.9] * len(images)

    @staticmethod
    def assert_return_prob(return_prob: bool) -> None:
        if not return_prob:
            raise AssertionError("VietOCR confidence must be requested.")


class VietOCRTextRecognizerTests(unittest.TestCase):
    def test_recognition_batch_size_bounds_vietocr_chunks(self) -> None:
        predictor = FakeVietOCRPredictor()
        recognizer = object.__new__(VietOCRTextRecognizer)
        recognizer._predictor = predictor
        images = [np.zeros((10, 30, 3), dtype=np.uint8) for _ in range(5)]

        results = recognizer.recognize(images, batch_size=2)

        self.assertEqual(predictor.chunk_sizes, [2, 2, 1])
        self.assertEqual(results, [RecognizedText("text", 0.9)] * 5)

    def test_fp16_recognition_enables_cuda_autocast(self) -> None:
        import torch

        predictor = FakeVietOCRPredictor()
        recognizer = object.__new__(VietOCRTextRecognizer)
        recognizer._predictor = predictor
        recognizer._use_fp16 = True
        recognizer._autocast_device_type = "cuda"
        images = [np.zeros((10, 30, 3), dtype=np.uint8)]

        with patch("torch.autocast", return_value=nullcontext()) as autocast:
            results = recognizer.recognize(images, batch_size=1)

        autocast.assert_called_once_with(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
        self.assertEqual(results, [RecognizedText("text", 0.9)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
