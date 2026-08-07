"""Typed objects used by the object-detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space bounding box in xyxy format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox":
        return BoundingBox(
            x_min=min(max(self.x_min, 0.0), float(image_width)),
            y_min=min(max(self.y_min, 0.0), float(image_height)),
            x_max=min(max(self.x_max, 0.0), float(image_width)),
            y_max=min(max(self.y_max, 0.0), float(image_height)),
        )

    def normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")

        clipped = self.clipped(image_width, image_height)
        return BoundingBox(
            x_min=clipped.x_min / image_width,
            y_min=clipped.y_min / image_height,
            x_max=clipped.x_max / image_width,
            y_max=clipped.y_max / image_height,
        )

    def to_xyxy(self, *, rounded: bool = False) -> list[float] | list[int]:
        values = [self.x_min, self.y_min, self.x_max, self.y_max]
        if rounded:
            return [int(round(value)) for value in values]
        return values

    def to_dict(self, *, rounded: bool = False) -> dict[str, float | int]:
        x_min, y_min, x_max, y_max = self.to_xyxy(rounded=rounded)
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame."""

    bbox: BoundingBox
    confidence: float
    class_index: int
    class_id: str
    class_name: str
    frame_id: str | None = None
    img_path: str | None = None
    model_name: str | None = None
    model_version: str | None = None

    def normalized_bbox(self, image_width: int, image_height: int) -> BoundingBox:
        return self.bbox.normalized(image_width, image_height)

    def to_dict(
        self,
        *,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "class_id": self.class_id,
            "class_index": self.class_index,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(rounded=True),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        if self.frame_id is not None:
            data["frame_id"] = self.frame_id
        if self.img_path is not None:
            data["img_path"] = self.img_path
        if image_width is not None and image_height is not None:
            data["bbox_norm"] = self.normalized_bbox(image_width, image_height).to_dict()
        return data


@dataclass(frozen=True, slots=True)
class FrameDetectionResult:
    """All object detections produced for a single image."""

    img_path: str
    detections: list[Detection]
    frame_id: str | None = None
    image_width: int | None = None
    image_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "img_path": self.img_path,
            "frame_id": self.frame_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detected_objects": [
                detection.to_dict(
                    image_width=self.image_width,
                    image_height=self.image_height,
                )
                for detection in self.detections
            ],
        }
