"""Domain exceptions raised by the clip extractor module."""


class ClipExtractorError(Exception):
    """Base exception for every clip extractor failure."""


class InvalidShotError(ClipExtractorError, ValueError):
    """Raised when an input Shot does not satisfy the data contract."""


class SourceVideoError(ClipExtractorError):
    """Raised when the source video is missing or cannot be inspected."""


class FFmpegNotAvailableError(ClipExtractorError):
    """Raised when FFmpeg/FFprobe is not available on the machine."""


class ClipMaterializationError(ClipExtractorError):
    """Raised when FFmpeg cannot create a requested clip file."""

