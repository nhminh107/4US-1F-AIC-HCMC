"""Các engine OCR có thể cắm vào ``OCRExtractor`` (module_ocr.md mục 5)."""

from BackEnd.app.ocr.engines.base import OCREngine, RawTextRegion
from BackEnd.app.ocr.engines.monkeyocr_engine import MonkeyOCREngine

__all__ = ["OCREngine", "RawTextRegion", "MonkeyOCREngine"]
