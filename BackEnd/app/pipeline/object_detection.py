"""Run object detection on persisted frame contracts and save results."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from BackEnd.CONFIG import PROJECT_ROOT
from BackEnd.app.contracts.pipeline import FrameMetadata, ObjectDetectionResult
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.openimages_jsonl import detect_frame


def detect_object(
    frame: FrameMetadata,
    db: PostgreManager,
    detector: Detector,
) -> list[ObjectDetectionResult]:
    """Detect objects in one frame and persist normalized contracts."""

    detections = detect_frame(
        _with_resolved_frame_path(frame),
        detector=detector,
    )
    results: list[ObjectDetectionResult] = []
    for detection in detections:
        record = db.add_object_detection(
            frame_id=detection.frame_id,
            class_id=detection.class_id,
            confidence=detection.confidence,
            x_min=detection.x_min,
            x_max=detection.x_max,
            y_min=detection.y_min,
            y_max=detection.y_max,
            model_name=detection.model_name,
            model_version=detection.model_version,
        )
        results.append(replace(detection, detection_id=record.detection_id))
    return results


def detect_objects(
    frames: list[FrameMetadata],
    db: PostgreManager,
    detector: Detector,
) -> list[ObjectDetectionResult]:
    """Detect and persist objects for a list of frames with one detector."""

    results: list[ObjectDetectionResult] = []
    for frame in frames:
        results.extend(detect_object(frame, db, detector))
    return results


def _with_resolved_frame_path(frame: FrameMetadata) -> FrameMetadata:
    """Resolve DB-relative paths without mutating the input contract."""

    if frame.frame_path is None:
        raise FileNotFoundError(
            f"Frame {frame.frame_id} has no local frame_path for object detection."
        )

    frame_path = Path(frame.frame_path)
    if not frame_path.is_absolute():
        frame_path = PROJECT_ROOT / frame_path
    if not frame_path.is_file():
        raise FileNotFoundError(f"Cannot read frame {frame.frame_id}: {frame_path}")
    return replace(frame, frame_path=frame_path)


__all__ = ["detect_object", "detect_objects"]
