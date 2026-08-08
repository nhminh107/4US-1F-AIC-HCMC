"""Run Open Images detection for one image and write contract records as JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.exporter import detections_to_contracts
from BackEnd.app.object_detection.postprocess import clip_detections
from BackEnd.app.object_detection.preprocess import load_image
from BackEnd.app.object_detection.tfhub_openimages_detector import TFHubOpenImagesDetector


def detect_image_array(
    image: np.ndarray,
    *,
    frame_id: str,
    detector: Detector,
    img_path: str | None = None,
) -> list[ObjectDetectionResult]:
    """Detect objects in a BGR image array and return contracts in memory."""
    image_height, image_width = image.shape[:2]
    detections = detector.detect(image, frame_id=frame_id, img_path=img_path)
    detections = clip_detections(
        detections,
        image_width=image_width,
        image_height=image_height,
    )
    return detections_to_contracts(
        detections,
        image_width=image_width,
        image_height=image_height,
        frame_id=frame_id,
    )


def _detect_image(
    image_path: Path,
    *,
    frame_id: str,
    detector: Detector | None,
) -> list[ObjectDetectionResult]:
    """Run detection for one image and return only in-memory pipeline contracts."""
    image = load_image(str(image_path))
    active_detector = detector or TFHubOpenImagesDetector()
    return detect_image_array(
        image,
        frame_id=frame_id,
        detector=active_detector,
        img_path=str(image_path),
    )


def detect_frame(
    frame: FrameMetadata,
    *,
    detector: Detector | None = None,
) -> list[ObjectDetectionResult]:
    """Detect objects for one frame and return contracts without writing files."""
    if frame.frame_path is None:
        raise ValueError("FrameMetadata.frame_path is required for object detection.")
    return _detect_image(
        frame.frame_path,
        frame_id=frame.frame_id,
        detector=detector,
    )


def detect_image_to_jsonl(
    image_path: Path,
    output_path: Path,
    *,
    detector: Detector | None = None,
    frame_id: str | None = None,
) -> list[ObjectDetectionResult]:
    """Detect objects in one image and write one pipeline contract per JSONL line."""
    resolved_frame_id = frame_id or image_path.stem
    contracts = _detect_image(
        image_path,
        frame_id=resolved_frame_id,
        detector=detector,
    )
    contracts = [
        replace(contract, detection_id=detection_id)
        for detection_id, contract in enumerate(contracts, start=1)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for contract in contracts:
            record = asdict(contract)
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")
    temporary_path.replace(output_path)
    return contracts


def detect_frame_to_jsonl(
    frame: FrameMetadata,
    output_path: Path,
    *,
    detector: Detector | None = None,
) -> list[ObjectDetectionResult]:
    """Run detection for a ``FrameMetadata`` input and write contract JSONL output."""
    contracts = detect_frame(frame, detector=detector)
    contracts = [
        replace(contract, detection_id=detection_id)
        for detection_id, contract in enumerate(contracts, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for contract in contracts:
            output_file.write(json.dumps(asdict(contract), ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")
    temporary_path.replace(output_path)
    return contracts
