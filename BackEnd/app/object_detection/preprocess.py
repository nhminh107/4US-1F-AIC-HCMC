"""Image loading and preprocessing for object detection."""

from __future__ import annotations

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for image preprocessing. "
            "Install dependencies with: uv sync --no-dev"
        ) from error
    return cv2


def load_image(image_path: str) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def resize_image(image: np.ndarray, max_side: int | None = None) -> np.ndarray:
    if max_side is None:
        return image
    if max_side <= 0:
        raise ValueError("max_side must be positive when provided.")

    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    cv2 = _cv2()
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def preprocess(image_path: str, max_side: int | None = None) -> np.ndarray:
    image = load_image(image_path)
    return resize_image(image, max_side=max_side)
