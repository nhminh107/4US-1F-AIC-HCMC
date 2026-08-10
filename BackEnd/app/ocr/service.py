"""Contract-first OCR orchestration for single frames and frame batches."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.ocr.config import OCRConfig
from BackEnd.app.ocr.engine import HybridOCREngine, OCREngine
from BackEnd.app.ocr.geometry import (
    normalize_polygon,
    normalized_bounding_box,
    perspective_crop,
    sort_regions_reading_order,
)
from BackEnd.app.ocr.schemas import DetectedTextRegion, PreparedTextRegion

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class OCRService:
    """Convert frame contracts into normalized OCR result contracts."""

    def __init__(
        self,
        config: OCRConfig | None = None,
        *,
        engine: OCREngine | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.config = config or OCRConfig()
        self._engine = engine
        self.project_root = project_root.resolve()

    def process_frame(self, frame: FrameMetadata) -> list[OCRResult]:
        """Run OCR for one frame contract."""

        if not isinstance(frame, FrameMetadata):
            raise TypeError("process_frame expects a FrameMetadata object.")
        return self.process_batch([frame])

    def process_batch(self, frames: list[FrameMetadata]) -> list[OCRResult]:
        """Run detection per image and recognition across one shared crop batch."""

        if not isinstance(frames, list):
            raise TypeError("process_batch expects list[FrameMetadata].")
        if not frames:
            return []
        if not all(isinstance(frame, FrameMetadata) for frame in frames):
            raise TypeError("process_batch expects only FrameMetadata objects.")

        frame_ids = [frame.frame_id for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("OCR batch contains duplicate frame_id values.")

        images = [self._load_frame_image(frame) for frame in frames]
        detections_by_frame = self.engine.detect(
            images,
            batch_size=self.config.detection_batch_size,
        )
        if len(detections_by_frame) != len(frames):
            raise RuntimeError("OCR engine returned an invalid detection batch size.")

        prepared_regions = self._prepare_regions(frames, images, detections_by_frame)
        recognition_results = self.engine.recognize(
            [region.cropped_image for region in prepared_regions],
            batch_size=self.config.recognition_batch_size,
        )
        if len(recognition_results) != len(prepared_regions):
            raise RuntimeError("OCR engine returned an invalid recognition batch size.")

        results_by_frame: list[list[OCRResult]] = [[] for _ in frames]
        for region, recognition in zip(prepared_regions, recognition_results):
            normalized_text = normalize_text(recognition.text)
            if (
                not normalized_text
                or recognition.confidence < self.config.recognition_score_threshold
            ):
                continue

            image_height, image_width = images[region.frame_index].shape[:2]
            x_min, x_max, y_min, y_max = normalized_bounding_box(
                region.polygon,
                image_width,
                image_height,
            )
            frame_results = results_by_frame[region.frame_index]
            frame_results.append(
                OCRResult(
                    frame_id=frames[region.frame_index].frame_id,
                    n=len(frame_results),
                    text=normalized_text,
                    language=self.config.language,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        return [result for frame_results in results_by_frame for result in frame_results]

    @property
    def engine(self) -> OCREngine:
        """Initialize the heavyweight model once, on first inference."""

        if self._engine is None:
            self._engine = HybridOCREngine(self.config)
            LOGGER.info(
                "Initialized OCR models detector=%s recognizer=%s version=%s",
                self.config.detection_model_name,
                self.config.recognition_model_name,
                self._engine.model_version,
            )
        return self._engine

    def close(self) -> None:
        """Release the model adapter if it exposes a close operation."""

        if self._engine is None:
            return
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    def _load_frame_image(self, frame: FrameMetadata) -> np.ndarray:
        if not frame.frame_id:
            raise ValueError("FrameMetadata.frame_id must not be empty.")
        if frame.frame_path is None:
            raise ValueError(f"Frame '{frame.frame_id}' has no local frame_path.")

        image_path = frame.frame_path.expanduser()
        if not image_path.is_absolute():
            image_path = self.project_root / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(
                f"Cannot read OCR frame '{frame.frame_id}' from: {image_path}"
            )

        image_height, image_width = image.shape[:2]
        if frame.width is not None and frame.width != image_width:
            raise ValueError(
                f"Frame '{frame.frame_id}' width mismatch: contract={frame.width}, "
                f"image={image_width}."
            )
        if frame.height is not None and frame.height != image_height:
            raise ValueError(
                f"Frame '{frame.frame_id}' height mismatch: contract={frame.height}, "
                f"image={image_height}."
            )
        return image

    def _prepare_regions(
        self,
        frames: Sequence[FrameMetadata],
        images: Sequence[np.ndarray],
        detections_by_frame: Sequence[list[DetectedTextRegion]],
    ) -> list[PreparedTextRegion]:
        prepared: list[PreparedTextRegion] = []
        for frame_index, (image, detections) in enumerate(
            zip(images, detections_by_frame)
        ):
            image_height, image_width = image.shape[:2]
            valid_regions: list[DetectedTextRegion] = []
            for detection in detections:
                if detection.confidence < self.config.detection_box_threshold:
                    continue
                try:
                    polygon = normalize_polygon(
                        detection.polygon,
                        image_width,
                        image_height,
                    )
                except ValueError:
                    LOGGER.warning(
                        "Skipping invalid OCR polygon for frame_id=%s.",
                        frames[frame_index].frame_id,
                    )
                    continue
                valid_regions.append(
                    DetectedTextRegion(
                        polygon=polygon,
                        confidence=detection.confidence,
                    )
                )

            for region in sort_regions_reading_order(valid_regions):
                try:
                    cropped_image = perspective_crop(
                        image,
                        region.polygon,
                        padding_ratio=self.config.crop_padding_ratio,
                        minimum_side=self.config.minimum_crop_side,
                    )
                except ValueError:
                    LOGGER.warning(
                        "Skipping unusable OCR crop for frame_id=%s.",
                        frames[frame_index].frame_id,
                    )
                    continue
                prepared.append(
                    PreparedTextRegion(
                        frame_index=frame_index,
                        polygon=region.polygon,
                        detection_confidence=region.confidence,
                        cropped_image=cropped_image,
                    )
                )
        return prepared


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without removing Vietnamese accents."""

    normalized = unicodedata.normalize("NFC", str(text))
    return " ".join(normalized.split())
