"""Export object-detection metadata to JSON and shared pipeline contracts."""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable

from BackEnd.app.contracts.pipeline import ClassMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.class_mapper import ClassMapper
from BackEnd.app.object_detection.schemas import Detection, FrameDetectionResult

FrameIdResolver = Callable[[str], str | None]


def export_results_json(
    results: Iterable[FrameDetectionResult],
    output_path: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = [result.to_dict() for result in results]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def detections_to_contracts(
    detections: Iterable[Detection],
    *,
    image_width: int,
    image_height: int,
    frame_id: str | None = None,
) -> list[ObjectDetectionResult]:
    contracts: list[ObjectDetectionResult] = []
    for detection in detections:
        resolved_frame_id = frame_id or detection.frame_id
        if not resolved_frame_id:
            raise ValueError("frame_id is required to build ObjectDetectionResult.")

        bbox = detection.normalized_bbox(image_width, image_height)
        contracts.append(
            ObjectDetectionResult(
                frame_id=resolved_frame_id,
                class_id=detection.class_id,
                confidence=detection.confidence,
                x_min=bbox.x_min,
                x_max=bbox.x_max,
                y_min=bbox.y_min,
                y_max=bbox.y_max,
                model_name=detection.model_name,
                model_version=detection.model_version,
            )
        )
    return contracts


def frame_results_to_contracts(
    results: Iterable[FrameDetectionResult],
    *,
    frame_id_resolver: FrameIdResolver | None = None,
) -> list[ObjectDetectionResult]:
    contracts: list[ObjectDetectionResult] = []
    for result in results:
        if result.image_width is None or result.image_height is None:
            raise ValueError(f"Missing image size for {result.img_path}.")
        frame_id = result.frame_id
        if frame_id is None and frame_id_resolver is not None:
            frame_id = frame_id_resolver(result.img_path)
        contracts.extend(
            detections_to_contracts(
                result.detections,
                image_width=result.image_width,
                image_height=result.image_height,
                frame_id=frame_id,
            )
        )
    return contracts


def class_metadata(mapper: ClassMapper) -> list[ClassMetadata]:
    return mapper.to_metadata()
