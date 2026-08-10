"""Contract-first OCR detection and recognition pipeline."""

from BackEnd.CONFIG import OCRConfig
from BackEnd.app.ocr.run_ocr_pipeline import run_ocr, run_ocr_batch
from BackEnd.app.ocr.service import OCRService

__all__ = ["OCRConfig", "OCRService", "run_ocr", "run_ocr_batch"]
