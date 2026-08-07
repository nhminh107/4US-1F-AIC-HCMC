"""Domain-specific embedding errors."""

from __future__ import annotations


class EmbeddingError(RuntimeError):
    """Base error for embedding pipeline failures."""


class MediaNotFoundError(EmbeddingError):
    """Raised when a local video asset cannot be resolved."""


class DecodeError(EmbeddingError):
    """Raised when media decoding fails."""


class ModelAdapterError(EmbeddingError):
    """Raised when model loading or inference fails."""

