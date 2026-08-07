"""Confidence, class filtering, and NMS helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from BackEnd.app.object_detection.class_mapper import ClassMapper
from BackEnd.app.object_detection.schemas import BoundingBox, Detection


def filter_by_confidence(
    detections: Iterable[Detection],
    confidence_threshold: float,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
    ]


def filter_by_classes(
    detections: Iterable[Detection],
    *,
    class_names: set[str] | None = None,
    class_ids: set[str] | None = None,
) -> list[Detection]:
    if not class_names and not class_ids:
        return list(detections)

    normalized_names = {name.strip().lower() for name in class_names or set()}
    allowed_ids = class_ids or set()
    return [
        detection
        for detection in detections
        if detection.class_id in allowed_ids
        or detection.class_name.strip().lower() in normalized_names
    ]


def clip_detections(
    detections: Iterable[Detection],
    *,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    clipped: list[Detection] = []
    for detection in detections:
        bbox = detection.bbox.clipped(image_width, image_height)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        clipped.append(replace(detection, bbox=bbox))
    return clipped


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)

    intersection_width = max(0.0, x_max - x_min)
    intersection_height = max(0.0, y_max - y_min)
    intersection = intersection_width * intersection_height
    union = box_a.area + box_b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def nms(
    detections: Iterable[Detection],
    iou_threshold: float = 0.45,
    *,
    class_aware: bool = True,
) -> list[Detection]:
    groups: dict[int, list[Detection]] = defaultdict(list)
    if class_aware:
        for detection in detections:
            groups[detection.class_index].append(detection)
    else:
        groups[0] = list(detections)

    kept: list[Detection] = []
    for group in groups.values():
        candidates = sorted(group, key=lambda item: item.confidence, reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                candidate
                for candidate in candidates
                if iou(best.bbox, candidate.bbox) < iou_threshold
            ]
    return kept


def resolve_class_indices(
    mapper: ClassMapper,
    *,
    class_names: list[str] | None = None,
    class_ids: list[str] | None = None,
) -> list[int] | None:
    if not class_names and not class_ids:
        return None

    indices: set[int] = set()
    for class_name in class_names or []:
        index = mapper.index_for_name(class_name)
        if index is not None:
            indices.add(index)
    for class_id in class_ids or []:
        index = mapper.index_for_class_id(class_id)
        if index is not None:
            indices.add(index)
    return sorted(indices)
