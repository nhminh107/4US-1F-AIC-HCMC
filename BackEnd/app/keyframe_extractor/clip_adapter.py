"""CLIP adapter bridge for hybrid keyframe selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from BackEnd.app.embedding.model_adapters.clip_vit_b32 import ClipViTB32Adapter


class ImageEmbeddingAdapter(Protocol):
    """Small image-only adapter surface needed by hybrid keyframe selection."""

    def encode_images(self, images: Sequence[object]) -> np.ndarray:
        ...


class ClipImageEmbeddingAdapter:
    """Image-only wrapper around the shared CLIP ViT-B/32 adapter."""

    def __init__(self, adapter: object | None = None, *, batch_size: int = 64) -> None:
        self.adapter = adapter or ClipViTB32Adapter()
        self.batch_size = int(batch_size)
        self._embedding_dim: int | None = None

    @property
    def model_name(self) -> str:
        metadata = self._metadata_or_none()
        if metadata is not None:
            return metadata.model_name
        return type(self.adapter).__name__

    @property
    def model_version(self) -> str | None:
        metadata = self._metadata_or_none()
        if metadata is not None:
            return metadata.model_revision
        return getattr(self.adapter, "model_revision", None)

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is not None:
            return self._embedding_dim
        get_dimension = getattr(self.adapter, "get_dimension", None)
        if callable(get_dimension):
            self._embedding_dim = int(get_dimension())
            return self._embedding_dim
        return 0

    def encode_images(self, images: Sequence[object]) -> np.ndarray:
        """Encode images and validate the returned embedding matrix."""

        values = list(images)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.asarray(
            self.adapter.encode_images(values, batch_size=self.batch_size),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(values) or vectors.shape[1] == 0:
            raise ValueError("CLIP adapter returned an invalid embedding matrix.")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("CLIP adapter returned NaN or infinity.")
        normalized = _normalize_rows(vectors)
        self._embedding_dim = int(normalized.shape[1])
        return normalized

    def _metadata_or_none(self):
        metadata = getattr(self.adapter, "metadata", None)
        if not callable(metadata):
            return None
        try:
            return metadata()
        except Exception:
            return None


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("CLIP adapter returned a zero vector.")
    return (vectors / norms).astype(np.float32)
