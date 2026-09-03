"""Small, replaceable text encoder wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    """Minimum interface required by the text-index builder."""

    model_id: str
    model_revision: str | None

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        """Return a two-dimensional matrix with one row per text."""


class SentenceTransformerTextEncoder:
    """Lazy SentenceTransformer encoder suitable for BGE-M3."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        *,
        model_revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id,
                revision=self.model_revision,
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)
