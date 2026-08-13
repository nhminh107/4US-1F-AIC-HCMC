"""OCR model adapters with batched detection and recognition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from BackEnd.CONFIG import OCRConfig
from BackEnd.app.ocr.schemas import DetectedTextRegion, RecognizedText


class OCREngine(Protocol):
    """Inference boundary used by the OCR service and test doubles."""

    @property
    def model_version(self) -> str:
        """Return a reproducible implementation version."""

    def detect(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[list[DetectedTextRegion]]:
        """Detect text polygons for each image in input order."""

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        """Recognize all rectified text crops in input order."""


class TextRecognitionBackend(Protocol):
    """Private recognition adapter shared by available backends."""

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        """Recognize crops in input order."""

    def close(self) -> None:
        """Release recognition model resources."""


class PaddleTextRecognizer:
    """Fast multilingual PaddleOCR recognition backend."""

    def __init__(
        self,
        config: OCRConfig,
        common_options: dict[str, object],
    ) -> None:
        from paddleocr import TextRecognition

        self._predictor = TextRecognition(
            model_name=config.recognition_model_name,
            model_dir=_optional_path_string(config.recognition_model_dir),
            **common_options,
        )

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        if not images:
            return []
        raw_results = self._predictor.predict(list(images), batch_size=batch_size)
        return [
            RecognizedText(
                text=str(result.get("rec_text", "")),
                confidence=float(result.get("rec_score", 0.0)),
            )
            for result in raw_results
        ]

    def close(self) -> None:
        self._predictor.close()


class VietOCRTextRecognizer:
    """Vietnamese-specialized batched recognition backend."""

    def __init__(self, config: OCRConfig, device: str) -> None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        if config.recognition_config_path is not None:
            vietocr_config = Cfg.load_config_from_file(
                str(config.recognition_config_path.expanduser().resolve())
            )
        else:
            vietocr_config = Cfg.load_config_from_name(config.recognition_model_name)
        torch_device = _paddle_to_torch_device(device)
        vietocr_config["device"] = torch_device
        if config.recognition_model_dir is not None:
            weights_path = config.recognition_model_dir.expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"VietOCR weight file does not exist: {weights_path}"
                )
            vietocr_config["weights"] = str(weights_path)

        self._predictor = Predictor(vietocr_config)
        self._predictor.model.eval()
        self._use_fp16 = config.precision == "fp16" and torch_device.startswith(
            "cuda"
        )
        self._autocast_device_type = torch_device.split(":", maxsplit=1)[0]

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        if not images:
            return []

        import torch
        from PIL import Image

        pil_images = [Image.fromarray(image[:, :, ::-1]) for image in images]
        recognized: list[RecognizedText] = []
        use_fp16 = getattr(self, "_use_fp16", False)
        autocast_device_type = getattr(self, "_autocast_device_type", "cuda")
        with torch.inference_mode(), torch.autocast(
            device_type=autocast_device_type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            for start in range(0, len(pil_images), batch_size):
                texts, probabilities = self._predictor.predict_batch(
                    pil_images[start : start + batch_size],
                    return_prob=True,
                )
                recognized.extend(
                    RecognizedText(text=str(text), confidence=float(probability))
                    for text, probability in zip(texts, probabilities)
                )
        return recognized

    def close(self) -> None:
        # VietOCR does not expose an explicit runtime close method.
        return None


class HybridOCREngine:
    """Paddle text detection plus a configurable batched recognizer."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        device = _resolve_device(config.device)

        # Lazy module imports keep contract/unit tests independent from model loading.
        from paddleocr import TextDetection

        common_options: dict[str, object] = {
            "device": device,
            "engine": config.engine,
            "enable_hpi": config.enable_hpi,
            "precision": config.precision,
            "cpu_threads": config.cpu_threads,
        }
        if (
            config.disable_new_ir
            and not config.enable_hpi
            and config.engine in {None, "paddle", "paddle_static"}
        ):
            common_options["engine_config"] = {
                "run_mode": "paddle",
                "enable_new_ir": False,
            }
        self._detector = TextDetection(
            model_name=config.detection_model_name,
            model_dir=_optional_path_string(config.detection_model_dir),
            limit_side_len=config.detection_limit_side_len,
            limit_type="max",
            thresh=config.detection_pixel_threshold,
            box_thresh=config.detection_box_threshold,
            unclip_ratio=config.detection_unclip_ratio,
            **common_options,
        )
        if config.recognition_backend == "vietocr":
            self._recognizer: TextRecognitionBackend = VietOCRTextRecognizer(
                config,
                device,
            )
        else:
            self._recognizer = PaddleTextRecognizer(config, common_options)

    @property
    def model_version(self) -> str:
        """Return the installed PaddleOCR package version."""

        try:
            paddle_version = version("paddleocr")
        except PackageNotFoundError:
            return self.config.model_version
        if self.config.recognition_backend == "vietocr":
            try:
                return f"paddleocr-{paddle_version}+vietocr-{version('vietocr')}"
            except PackageNotFoundError:
                pass
        return f"paddleocr-{paddle_version}"

    def detect(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[list[DetectedTextRegion]]:
        """Run batched PaddleOCR text detection."""

        if not images:
            return []
        raw_results = self._detector.predict(list(images), batch_size=batch_size)
        detections: list[list[DetectedTextRegion]] = []
        for result in raw_results:
            polygons = list(result.get("dt_polys", []))
            scores = list(result.get("dt_scores", []))
            if len(polygons) != len(scores):
                raise RuntimeError("PaddleOCR returned mismatched polygons and scores.")
            detections.append(
                [
                    DetectedTextRegion(
                        polygon=np.asarray(polygon, dtype=np.float32),
                        confidence=float(score),
                    )
                    for polygon, score in zip(polygons, scores)
                ]
            )
        if len(detections) != len(images):
            raise RuntimeError(
                "PaddleOCR detection result count does not match the input batch."
            )
        return detections

    def recognize(
        self,
        images: Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> list[RecognizedText]:
        """Run recognition over crops from the entire frame batch."""

        if not images:
            return []
        recognized = self._recognizer.recognize(images, batch_size=batch_size)
        if len(recognized) != len(images):
            raise RuntimeError(
                "OCR recognition result count does not match the crop batch."
            )
        return recognized

    def close(self) -> None:
        """Release model runtimes when supported by PaddleOCR."""

        self._detector.close()
        self._recognizer.close()


def _resolve_device(requested_device: str | None) -> str:
    """Resolve automatic execution while rejecting unavailable explicit GPU use."""

    import paddle

    cuda_available = bool(
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    )
    if requested_device is not None:
        if requested_device.startswith("gpu") and not cuda_available:
            raise RuntimeError(
                f"OCR device '{requested_device}' was requested, but PaddlePaddle "
                "was built without CUDA support."
            )
        return requested_device
    return "gpu:0" if cuda_available else "cpu"


def _optional_path_string(path: Path | None) -> str | None:
    return str(path.expanduser().resolve()) if path is not None else None


def _paddle_to_torch_device(device: str) -> str:
    if device == "gpu":
        return "cuda"
    if device.startswith("gpu:"):
        return f"cuda:{device.split(':', maxsplit=1)[1]}"
    return device


# Backward-compatible internal name for code written during the initial rewrite.
PaddleOCREngine = HybridOCREngine
