"""Utilities for scanning keyframes and visualizing detections."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from BackEnd.CONFIG import OBJECT_DETECTION_IMAGE_EXTENSIONS as IMAGE_EXTENSIONS
from BackEnd.app.object_detection.schemas import Detection


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for drawing detections. "
            "Install dependencies with: uv sync --no-dev"
        ) from error
    return cv2


def scan_keyframes(keyframes_dir: str) -> list[str]:
    img_paths: list[str] = []
    for root, _dirs, files in os.walk(keyframes_dir):
        for filename in files:
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            rel_path = os.path.relpath(os.path.join(root, filename), keyframes_dir)
            img_paths.append(rel_path.replace(os.sep, "/"))
    img_paths.sort()
    return img_paths


def iter_chunks(items: list[str], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def default_frame_id(img_path: str) -> str:
    """Best-effort frame id for standalone JSON output.

    Database insertion should pass the real Frame.frame_id instead.
    """

    return Path(img_path).stem


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    *,
    color: tuple[int, int, int] = (0, 180, 255),
) -> np.ndarray:
    cv2 = _cv2()
    canvas = image.copy()
    for detection in detections:
        x_min, y_min, x_max, y_max = detection.bbox.to_xyxy(rounded=True)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x_min, max(16, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_annotated_image(
    image: np.ndarray,
    detections: Iterable[Detection],
    output_path: str,
) -> None:
    cv2 = _cv2()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, draw_detections(image, detections))
