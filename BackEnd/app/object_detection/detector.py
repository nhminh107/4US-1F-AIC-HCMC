"""Detector interface shared by object-detection implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from BackEnd.app.object_detection.schemas import Detection


class Detector(ABC):
    """Base interface for image object detectors."""

    model_name: str
    model_version: str | None

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: str | None = None,
        img_path: str | None = None,
    ) -> list[Detection]:
        """Run detection on an OpenCV image and return normalized objects."""

    def detect_batch(
        self,
        images: list[np.ndarray],
        *,
        frame_ids: list[str | None] | None = None,
        img_paths: list[str | None] | None = None,
    ) -> list[list[Detection]]:
        """Detect a batch, with a safe sequential fallback for implementations."""

        resolved_frame_ids = (
            frame_ids if frame_ids is not None else [None] * len(images)
        )
        resolved_img_paths = (
            img_paths if img_paths is not None else [None] * len(images)
        )
        if len(resolved_frame_ids) != len(images):
            raise ValueError("frame_ids must contain one value per image.")
        if len(resolved_img_paths) != len(images):
            raise ValueError("img_paths must contain one value per image.")
        return [
            self.detect(image, frame_id=frame_id, img_path=img_path)
            for image, frame_id, img_path in zip(
                images,
                resolved_frame_ids,
                resolved_img_paths,
            )
        ]
