"""Configuration for the OCR inference service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIETOCR_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "vietocr_vgg_transformer.yml"
)


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Reproducible OCR model and inference settings."""

    detection_model_name: str = "PP-OCRv5_mobile_det"
    recognition_backend: str = "vietocr"
    recognition_model_name: str = "vgg_transformer"
    detection_model_dir: Path | None = None
    recognition_model_dir: Path | None = None
    recognition_config_path: Path | None = DEFAULT_VIETOCR_CONFIG_PATH
    model_version: str = "paddleocr-3.x"
    language: str = "vi"
    device: str | None = None
    engine: str | None = None
    enable_hpi: bool = False
    disable_new_ir: bool = True
    precision: str = "fp32"
    cpu_threads: int = 8
    detection_batch_size: int = 4
    recognition_batch_size: int = 32
    detection_limit_side_len: int = 1280
    detection_pixel_threshold: float = 0.3
    detection_box_threshold: float = 0.5
    detection_unclip_ratio: float = 1.6
    recognition_score_threshold: float = 0.45
    crop_padding_ratio: float = 0.04
    minimum_crop_side: int = 3

    def __post_init__(self) -> None:
        """Reject invalid settings before model initialization."""

        if not self.detection_model_name or not self.recognition_model_name:
            raise ValueError("OCR model names must not be empty.")
        if self.recognition_backend not in {"paddleocr", "vietocr"}:
            raise ValueError(
                "recognition_backend must be either 'paddleocr' or 'vietocr'."
            )
        if not self.language:
            raise ValueError("OCR language must not be empty.")
        if self.detection_batch_size <= 0 or self.recognition_batch_size <= 0:
            raise ValueError("OCR batch sizes must be positive.")
        if self.detection_limit_side_len <= 0:
            raise ValueError("detection_limit_side_len must be positive.")
        if self.minimum_crop_side <= 0:
            raise ValueError("minimum_crop_side must be positive.")
        if self.crop_padding_ratio < 0:
            raise ValueError("crop_padding_ratio must be non-negative.")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive.")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be either 'fp32' or 'fp16'.")

        thresholds = {
            "detection_pixel_threshold": self.detection_pixel_threshold,
            "detection_box_threshold": self.detection_box_threshold,
            "recognition_score_threshold": self.recognition_score_threshold,
        }
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")
