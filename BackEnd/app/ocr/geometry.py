"""Geometry helpers that preserve OCR polygon alignment."""

from __future__ import annotations

import cv2
import numpy as np

from BackEnd.app.ocr.schemas import DetectedTextRegion


def normalize_polygon(polygon: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Return a clipped four-point polygon in TL, TR, BR, BL order."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("OCR polygon must contain at least four finite points.")
    if len(points) != 4:
        rectangle = cv2.minAreaRect(points)
        points = cv2.boxPoints(rectangle)

    points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[2] = points[np.argmax(sums)]
    ordered[3] = points[np.argmax(differences)]

    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("OCR polygon collapses after clipping.")
    return ordered


def expand_polygon(
    polygon: np.ndarray,
    padding_ratio: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Expand a polygon around its center and clip it to the image."""

    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative.")
    center = polygon.mean(axis=0, keepdims=True)
    expanded = center + (polygon - center) * (1.0 + 2.0 * padding_ratio)
    return normalize_polygon(expanded, image_width, image_height)


def perspective_crop(
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    padding_ratio: float = 0.0,
    minimum_side: int = 3,
) -> np.ndarray:
    """Rectify a quadrilateral text polygon without writing an intermediate file."""

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("OCR image must be a non-empty BGR image with three channels.")

    image_height, image_width = image.shape[:2]
    ordered = normalize_polygon(polygon, image_width, image_height)
    if padding_ratio:
        ordered = expand_polygon(ordered, padding_ratio, image_width, image_height)

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    target_width = int(round(max(top_width, bottom_width)))
    target_height = int(round(max(left_height, right_height)))
    if target_width < minimum_side or target_height < minimum_side:
        raise ValueError("OCR polygon is too small to recognize.")

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalized_bounding_box(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert a pixel polygon to normalized x_min, x_max, y_min, y_max."""

    ordered = normalize_polygon(polygon, image_width, image_height)
    x_values = ordered[:, 0]
    y_values = ordered[:, 1]
    x_min = float(x_values.min() / image_width)
    x_max = float(x_values.max() / image_width)
    y_min = float(y_values.min() / image_height)
    y_max = float(y_values.max() / image_height)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("OCR polygon has no positive normalized area.")
    return x_min, x_max, y_min, y_max


def sort_regions_reading_order(regions: list[DetectedTextRegion]) -> list[DetectedTextRegion]:
    """Sort regions top-to-bottom, then left-to-right within a text line."""

    if not regions:
        return []

    heights = [
        max(
            np.linalg.norm(region.polygon[3] - region.polygon[0]),
            np.linalg.norm(region.polygon[2] - region.polygon[1]),
        )
        for region in regions
    ]
    line_tolerance = max(8.0, float(np.median(heights)) * 0.5)

    def sort_key(region: DetectedTextRegion) -> tuple[int, float]:
        center_x = float(region.polygon[:, 0].mean())
        center_y = float(region.polygon[:, 1].mean())
        return round(center_y / line_tolerance), center_x

    return sorted(regions, key=sort_key)
