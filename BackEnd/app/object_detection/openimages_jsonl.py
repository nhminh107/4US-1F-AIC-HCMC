"""Run Open Images detection for one image and write contract records as JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import lru_cache
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
    active_detector = detector or _default_detector()
    return detect_image_array(
        image,
        frame_id=frame_id,
        detector=active_detector,
        img_path=str(image_path),
    )


@lru_cache(maxsize=1)
def _default_detector() -> TFHubOpenImagesDetector:
    """Load and retain the default Open Images model once per process."""

    return TFHubOpenImagesDetector()


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


def detect_frames(
    frames: list[FrameMetadata],
    *,
    detector: Detector | None = None,
    batch_size: int | None = None,
) -> list[ObjectDetectionResult]:
    """Detect a frame batch while loading each image and model only once."""

    if not frames:
        return []
    active_detector = detector or _default_detector()
    resolved_batch_size = batch_size or getattr(active_detector, "batch_size", 1)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    contracts: list[ObjectDetectionResult] = []
    for start in range(0, len(frames), resolved_batch_size):
        frame_batch = frames[start : start + resolved_batch_size]
        images: list[np.ndarray] = []
        for frame in frame_batch:
            if frame.frame_path is None:
                raise ValueError(
                    f"FrameMetadata.frame_path is required for {frame.frame_id}."
                )
            images.append(load_image(str(frame.frame_path)))

        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for index, image in enumerate(images):
            shape_groups.setdefault(image.shape, []).append(index)

        contracts_by_index: dict[int, list[ObjectDetectionResult]] = {}
        for indices in shape_groups.values():
            grouped_images = [images[index] for index in indices]
            grouped_frames = [frame_batch[index] for index in indices]
            detection_batches = active_detector.detect_batch(
                grouped_images,
                frame_ids=[frame.frame_id for frame in grouped_frames],
                img_paths=[str(frame.frame_path) for frame in grouped_frames],
            )
            for batch_index, frame, image, detections in zip(
                indices,
                grouped_frames,
                grouped_images,
                detection_batches,
            ):
                clipped = clip_detections(
                    detections,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                )
                contracts_by_index[batch_index] = detections_to_contracts(
                    clipped,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                    frame_id=frame.frame_id,
                )
        for index in range(len(frame_batch)):
            contracts.extend(contracts_by_index[index])
    return contracts


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
