"""Runtime protocols for embedding service dependencies."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from BackEnd.CONFIG import EMBEDDING_BATCH_SIZE

from BackEnd.app.contracts.embedding import DecodedFrameBatch, ModelMetadata, VideoAsset


class ImageTextEmbeddingAdapter(Protocol):
    """Minimal adapter interface consumed by embedding services."""

    @property
    def embedding_space_id(self) -> str:
        ...

    def encode_images(
        self, images, batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        ...

    def encode_texts(
        self, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        ...

    def get_dimension(self) -> int:
        ...

    def metadata(self) -> ModelMetadata:
        ...


class VideoDecoder(Protocol):
    """Minimal decoder interface consumed by clip embedding."""

    def decode_nearest_frames(
        self,
        video_asset: VideoAsset,
        timestamps_ms: list[int] | tuple[int, ...],
    ) -> DecodedFrameBatch:
        ...
