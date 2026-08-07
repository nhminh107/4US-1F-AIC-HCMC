"""Detector interface shared by object-detection implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from BackEnd.app.object_detection.class_mapper import ClassMapper
from BackEnd.app.object_detection.schemas import Detection


class Detector(ABC):
    """Base interface for image object detectors."""

    model_name: str
    model_version: str | None
    class_mapper: ClassMapper

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: str | None = None,
        img_path: str | None = None,
    ) -> list[Detection]:
        """Run detection on an OpenCV image and return normalized objects."""
