"""Parse model-specific outputs into Detection objects."""

from __future__ import annotations

from typing import Any

from BackEnd.app.object_detection.class_mapper import ClassMapper
from BackEnd.app.object_detection.schemas import BoundingBox, Detection


def _to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def parse_yolo_result(
    result: Any,
    *,
    class_mapper: ClassMapper,
    frame_id: str | None = None,
    img_path: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
) -> list[Detection]:
    """Parse one ultralytics result object."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = _to_list(boxes.xyxy)
    confidence_values = _to_list(boxes.conf)
    class_values = _to_list(boxes.cls)

    detections: list[Detection] = []
    for xyxy, confidence, class_index_value in zip(
        xyxy_values,
        confidence_values,
        class_values,
    ):
        class_index = int(class_index_value)
        detections.append(
            Detection(
                bbox=BoundingBox(
                    x_min=float(xyxy[0]),
                    y_min=float(xyxy[1]),
                    x_max=float(xyxy[2]),
                    y_max=float(xyxy[3]),
                ),
                confidence=float(confidence),
                class_index=class_index,
                class_id=class_mapper.class_id_for_index(class_index),
                class_name=class_mapper.name_for_index(class_index),
                frame_id=frame_id,
                img_path=img_path,
                model_name=model_name,
                model_version=model_version,
            )
        )
    return detections
