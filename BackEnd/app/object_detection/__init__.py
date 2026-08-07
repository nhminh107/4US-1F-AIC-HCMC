"""Object detection module for keyframe analysis."""

from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.schemas import BoundingBox, Detection, FrameDetectionResult
from BackEnd.app.object_detection.yolo_detector import YOLODetector

__all__ = [
    "BoundingBox",
    "Detection",
    "Detector",
    "FrameDetectionResult",
    "YOLODetector",
]
