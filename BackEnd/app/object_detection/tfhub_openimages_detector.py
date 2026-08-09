"""TensorFlow Hub Open Images detector backed by Faster R-CNN Inception-ResNet-v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
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
        batch_size: int = 8,
        device: str | None = None,
        expected_model_sha256: str | None = None,
        require_local_model: bool = False,
        supports_batching: bool | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.device = device
        self.model_source = model_url
        self.model_weights_hash: str | None = None
        self._tensorflow: Any | None = None

        if model is None:
            self._tensorflow, model, self.model_weights_hash = self._load_model(
                model_url,
                expected_model_sha256=expected_model_sha256,
                require_local_model=require_local_model,
            )
        if self.model_weights_hash is not None:
            self.model_version = (
                f"{MODEL_VERSION}+sha256:{self.model_weights_hash[:12]}"
            )
        self.model = model
        self.supports_batching = (
            _signature_supports_batching(model)
            if supports_batching is None
            else supports_batching
        )

    @staticmethod
    def _load_model(
        model_source: str,
        *,
        expected_model_sha256: str | None,
        require_local_model: bool,
    ) -> tuple[Any, Any, str | None]:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as error:
            raise ImportError(
                "TFHubOpenImagesDetector requires tensorflow and tensorflow-hub. "
                "Install them with: python -m pip install 'tensorflow>=2.16,<2.19' "
                "'tensorflow-hub>=0.16,<0.17'"
            ) from error
        source_path = Path(model_source).expanduser()
        is_local = source_path.exists()
        if require_local_model and not is_local:
            raise FileNotFoundError(
                "A local Open Images model is required but was not found: "
                f"{source_path}"
            )

        model_hash = None
        if expected_model_sha256 is not None:
            if not is_local:
                raise ValueError(
                    "expected_model_sha256 can only verify a local model path."
                )
            model_hash = _sha256_path(source_path)
            if model_hash.lower() != expected_model_sha256.lower():
                raise ValueError(
                    "Open Images model checksum mismatch: "
                    f"expected {expected_model_sha256}, got {model_hash}"
                )
        elif is_local:
            model_hash = _sha256_path(source_path)

        resolved_source = str(source_path.resolve()) if is_local else model_source
        loaded_model = hub.load(resolved_source)
        signatures = getattr(loaded_model, "signatures", {})
        for signature_name in ("default", "serving_default"):
            if signature_name in signatures:
                return tf, signatures[signature_name], model_hash
        raise ValueError(
            "The TF Hub model has no supported inference signature. "
            f"Available signatures: {sorted(signatures)}"
        )

    def _predict_batch(self, rgb_images: np.ndarray) -> Mapping[str, Any]:
        model_input: Any = rgb_images
        if self._tensorflow is not None:
            uint8_images = self._tensorflow.convert_to_tensor(
                rgb_images,
                dtype=self._tensorflow.uint8,
            )
            model_input = self._tensorflow.image.convert_image_dtype(
                uint8_images,
                self._tensorflow.float32,
            )
        device_context = (
            self._tensorflow.device(self.device)
            if self._tensorflow is not None and self.device is not None
            else nullcontext()
        )
        with device_context:
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
        return self.detect_batch(
            [image],
            frame_ids=[frame_id],
            img_paths=[img_path],
        )[0]

    def detect_batch(
        self,
        images: list[np.ndarray],
        *,
        frame_ids: list[str | None] | None = None,
        img_paths: list[str | None] | None = None,
    ) -> list[list[Detection]]:
        """Run one TensorFlow call for a same-shaped image batch."""

        if not images:
            return []
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

        first_shape = images[0].shape
        for image in images:
            self._validate_image(image)
            if image.shape != first_shape:
                raise ValueError("All images in a detection batch must have the same shape.")

        if len(images) > 1 and not self.supports_batching:
            return [
                self._detect_one(image, frame_id=frame_id, img_path=img_path)
                for image, frame_id, img_path in zip(
                    images,
                    resolved_frame_ids,
                    resolved_img_paths,
                )
            ]

        rgb_images = np.stack([image[:, :, ::-1] for image in images])
        outputs = self._predict_batch(rgb_images)
        per_image_outputs = _split_batch_outputs(outputs, len(images))
        return [
            self._parse_outputs(
                image_outputs,
                image_width=image.shape[1],
                image_height=image.shape[0],
                frame_id=frame_id,
                img_path=img_path,
            )
            for image_outputs, image, frame_id, img_path in zip(
                per_image_outputs,
                images,
                resolved_frame_ids,
                resolved_img_paths,
            )
        ]

    def _detect_one(
        self,
        image: np.ndarray,
        *,
        frame_id: str | None,
        img_path: str | None,
    ) -> list[Detection]:
        outputs = self._predict_batch(np.stack([image[:, :, ::-1]]))
        image_outputs = _split_batch_outputs(outputs, 1)[0]
        return self._parse_outputs(
            image_outputs,
            image_width=image.shape[1],
            image_height=image.shape[0],
            frame_id=frame_id,
            img_path=img_path,
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [height, width, 3] in BGR format.")

    def _parse_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        image_width: int,
        image_height: int,
        frame_id: str | None,
        img_path: str | None,
    ) -> list[Detection]:
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


def _split_batch_outputs(
    outputs: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Split model output arrays into one mapping per input image."""

    converted = {key: _to_list(value) for key, value in outputs.items()}
    if batch_size == 1:
        return [
            {
                key: _unwrap_single_batch_axis(key, value)
                for key, value in converted.items()
            }
        ]

    for key, value in converted.items():
        if len(value) != batch_size:
            raise ValueError(
                f"Model output '{key}' does not have batch size {batch_size}."
            )
    return [
        {key: value[index] for key, value in converted.items()}
        for index in range(batch_size)
    ]


def _unwrap_single_batch_axis(field_name: str, value: list[Any]) -> list[Any]:
    if len(value) != 1 or not isinstance(value[0], list):
        return value
    first = value[0]
    if field_name == "detection_boxes":
        return first if first and isinstance(first[0], list) else value
    return first if first and isinstance(first[0], list) else value


def _sha256_path(path: Path) -> str:
    """Hash a local SavedModel file or directory deterministically."""

    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    if not files:
        raise ValueError(f"Local model path contains no files: {path}")
    for file_path in files:
        relative_name = (
            file_path.name
            if path.is_file()
            else file_path.relative_to(path).as_posix()
        )
        digest.update(relative_name.encode("utf-8"))
        with file_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _signature_supports_batching(model: Any) -> bool:
    """Return whether a loaded TensorFlow signature accepts batch size > 1."""

    signature = getattr(model, "structured_input_signature", None)
    if signature is None:
        return True
    positional, keyword = signature
    specs = list(positional) + list(keyword.values())
    for spec in specs:
        shape = getattr(spec, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        batch_dimension = shape[0]
        if batch_dimension is None:
            return True
        return int(batch_dimension) > 1
    return False
