"""YOLO implementation of the Detector interface."""

from __future__ import annotations

import numpy as np

from BackEnd.app.object_detection.class_mapper import ClassMapper
from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.parser import parse_yolo_result
from BackEnd.app.object_detection.postprocess import resolve_class_indices
from BackEnd.app.object_detection.schemas import Detection


class YOLODetector(Detector):
    """Object detector backed by ultralytics YOLO."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        class_names: list[str] | None = None,
        class_ids: list[str] | None = None,
        max_det: int = 300,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install dependencies with: uv sync --no-dev"
            ) from error

        self.model = YOLO(model_path)
        model_names = getattr(self.model, "names", None)
        self.class_mapper = ClassMapper(model_names)
        self.model_name = "YOLO"
        self.model_version = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_det = max_det
        self.class_indices = resolve_class_indices(
            self.class_mapper,
            class_names=class_names,
            class_ids=class_ids,
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: str | None = None,
        img_path: str | None = None,
    ) -> list[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.class_indices,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []

        return parse_yolo_result(
            results[0],
            class_mapper=self.class_mapper,
            frame_id=frame_id,
            img_path=img_path,
            model_name=self.model_name,
            model_version=self.model_version,
        )
