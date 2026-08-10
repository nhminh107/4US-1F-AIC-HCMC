"""TensorFlow Hub Open Images detector backed by Faster R-CNN Inception-ResNet-v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.schemas import BoundingBox, Detection


MODEL_URL = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
MODEL_NAME = "faster_rcnn/inception_resnet_v2"
MODEL_VERSION = "openimages_v4/1"


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap TF Hub's common ``(N, 1)`` scalar output values."""
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected a scalar model output value, got {value!r}.")
        value = value[0]
    return value


class TFHubOpenImagesDetector(Detector):
    """Run the Open Images Faster R-CNN model and return pixel-space detections.

    The TensorFlow Hub model emits normalized boxes in ``[y_min, x_min,
    y_max, x_max]`` order. This adapter converts them to the internal xyxy
    representation while preserving the Open Images MID as ``class_id``.
    """

    def __init__(
        self,
        *,
        model_url: str = MODEL_URL,
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION
        self.confidence_threshold = confidence_threshold
        self._tensorflow: Any | None = None

        if model is None:
            self._tensorflow, model = self._load_model(model_url)
        self.model = model

    @staticmethod
    def _load_model(model_url: str) -> tuple[Any, Any]:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as error:
            raise ImportError(
                "TFHubOpenImagesDetector requires tensorflow and tensorflow-hub. "
                "Install them with: python -m pip install 'tensorflow>=2.16,<2.19' "
                "'tensorflow-hub>=0.16,<0.17'"
            ) from error
        loaded_model = hub.load(model_url)
        signatures = getattr(loaded_model, "signatures", {})
        for signature_name in ("default", "serving_default"):
            if signature_name in signatures:
                return tf, signatures[signature_name]
        raise ValueError(
            "The TF Hub model has no supported inference signature. "
            f"Available signatures: {sorted(signatures)}"
        )

    def _predict(self, rgb_image: np.ndarray) -> Mapping[str, Any]:
        model_input: Any = rgb_image
        if self._tensorflow is not None:
            uint8_image = self._tensorflow.convert_to_tensor(
                rgb_image,
                dtype=self._tensorflow.uint8,
            )
            model_input = self._tensorflow.image.convert_image_dtype(
                uint8_image,
                self._tensorflow.float32,
            )[self._tensorflow.newaxis, ...]
        outputs = self.model(model_input)
        if not isinstance(outputs, Mapping):
            raise TypeError("The TF Hub detector must return a mapping of detection tensors.")
        return outputs

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_id: str | None = None,
        img_path: str | None = None,
    ) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [height, width, 3] in BGR format.")

        image_height, image_width = image.shape[:2]
        rgb_image = image[:, :, ::-1]
        outputs = self._predict(rgb_image)
        required_fields = (
            "detection_boxes",
            "detection_scores",
            "detection_class_names",
            "detection_class_entities",
            "detection_class_labels",
        )
        missing_fields = [field for field in required_fields if field not in outputs]
        if missing_fields:
            raise ValueError(f"Model output is missing fields: {', '.join(missing_fields)}")

        boxes = _to_list(outputs["detection_boxes"])
        scores = _to_list(outputs["detection_scores"])
        class_mids = _to_list(outputs["detection_class_names"])
        class_names = _to_list(outputs["detection_class_entities"])
        class_labels = _to_list(outputs["detection_class_labels"])
        lengths = {len(boxes), len(scores), len(class_mids), len(class_names), len(class_labels)}
        if len(lengths) != 1:
            raise ValueError("Model output arrays have inconsistent lengths.")

        detections: list[Detection] = []
        for box, score, class_mid, class_name, class_label in zip(
            boxes,
            scores,
            class_mids,
            class_names,
            class_labels,
        ):
            confidence = float(_unwrap_scalar(score))
            if confidence < self.confidence_threshold:
                continue
            if len(box) != 4:
                raise ValueError(f"Expected a four-value bounding box, got {box!r}.")

            y_min, x_min, y_max, x_max = (float(value) for value in box)
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x_min=x_min * image_width,
                        y_min=y_min * image_height,
                        x_max=x_max * image_width,
                        y_max=y_max * image_height,
                    ),
                    confidence=confidence,
                    class_index=int(_unwrap_scalar(class_label)),
                    class_id=_decode_text(_unwrap_scalar(class_mid)),
                    class_name=_decode_text(_unwrap_scalar(class_name)),
                    frame_id=frame_id,
                    img_path=img_path,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return detections
