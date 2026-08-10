"""Tests for centralized backend configuration and legacy imports."""

from BackEnd import CONFIG
from BackEnd.app.clip_extractor import ClipExtractorConfig
from BackEnd.app.embedding import CONFIG as legacy_embedding_config
from BackEnd.app.ocr.config import OCRConfig as LegacyOCRConfig
from BackEnd.app.tracking.CONFIG import TrackingConfig as LegacyTrackingConfig


def test_module_batch_sizes_preserve_existing_values() -> None:
    assert CONFIG.GENERAL_BATCH_SIZE == 32
    assert CONFIG.general_batch_size == CONFIG.GENERAL_BATCH_SIZE
    assert CONFIG.EMBEDDING_BATCH_SIZE == 64
    assert CONFIG.OCR_DETECTION_BATCH_SIZE == 4
    assert CONFIG.OCR_RECOGNITION_BATCH_SIZE == 32
    assert CONFIG.SHOT_WINDOW_BATCH_SIZE == 8
    assert CONFIG.OBJECT_DETECTION_CHUNK_SIZE == 100
    assert CONFIG.ELASTICSEARCH_BULK_BATCH_SIZE == 500
    assert CONFIG.ASR_TOTAL_BATCH_DURATION_SECONDS == 1_800


def test_legacy_config_imports_reexport_central_classes() -> None:
    assert legacy_embedding_config.ClipBuilderConfig is CONFIG.ClipBuilderConfig
    assert legacy_embedding_config.batch_size == CONFIG.EMBEDDING_BATCH_SIZE
    assert LegacyOCRConfig is CONFIG.OCRConfig
    assert LegacyTrackingConfig is CONFIG.TrackingConfig
    assert ClipExtractorConfig is CONFIG.ClipExtractorConfig
