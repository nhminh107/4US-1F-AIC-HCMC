"""Public API for the Clip Extractor module."""

from .clip_extractor import ClipExtractor, ClipExtractorConfig, plan_clip_windows
from .contracts import ClipRecord, ShotRecord
from .exceptions import (
    ClipExtractorError,
    ClipMaterializationError,
    FFmpegNotAvailableError,
    InvalidShotError,
    SourceVideoError,
)

__all__ = [
    "ClipExtractor",
    "ClipExtractorConfig",
    "plan_clip_windows",
    "ClipRecord",
    "ShotRecord",
    "ClipExtractorError",
    "ClipMaterializationError",
    "FFmpegNotAvailableError",
    "InvalidShotError",
    "SourceVideoError",
]
