"""Public OCR entry points that use shared pipeline contracts."""

from __future__ import annotations

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.ocr.service import OCRService

_DEFAULT_SERVICE: OCRService | None = None


def run_ocr(
    frame: FrameMetadata,
    *,
    service: OCRService | None = None,
) -> list[OCRResult]:
    """Run OCR for a single frame contract."""

    return (service or get_default_service()).process_frame(frame)


def run_ocr_batch(
    frames: list[FrameMetadata],
    *,
    service: OCRService | None = None,
) -> list[OCRResult]:
    """Run OCR for multiple frame contracts with shared model batches."""

    return (service or get_default_service()).process_batch(frames)


def get_default_service() -> OCRService:
    """Return one lazily initialized OCR service for the current process."""

    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = OCRService()
    return _DEFAULT_SERVICE
