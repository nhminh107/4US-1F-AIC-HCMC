"""CLIP ViT-B/32 model adapter implementation."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from BackEnd.app.contracts.embedding import ModelMetadata
from BackEnd.app.embedding import CONFIG


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector về độ dài L2 = 1."""
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class ClipViTB32Adapter:
    """Adapter wrapper cho model sentence-transformers/clip-ViT-B-32."""

    def __init__(
        self,
        model: object | None = None,
        *,
        dimension: int = CONFIG.CLIP_DIMENSION,
        device: object | None = None,
    ) -> None:
        self._model = model
        self.dimension = int(dimension)
        self.device = device or CONFIG.device
        self.embedding_space_id = getattr(CONFIG, "CLIP_EMBEDDING_SPACE_ID", "clip.clip_vit_b32.masked_mean16_v1")

    @property
    def model(self):
        if self._model is None:
            from pathlib import Path
            from sentence_transformers import SentenceTransformer
            
            project_root = Path(__file__).resolve().parents[4]
            local_model_path = project_root / "data" / "models" / "clip-ViT-B-32"
            
            model_target = str(local_model_path) if local_model_path.exists() else CONFIG.CLIP_MODEL
            self._model = SentenceTransformer(model_target, device=str(self.device))
        return self._model

    @property
    def model_name(self) -> str:
        return CONFIG.CLIP_MODEL

    @property
    def model_revision(self) -> str | None:
        return CONFIG.CLIP_MODEL_REVISION

    def get_dimension(self) -> int:
        return self.dimension

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_backend=getattr(CONFIG, "CLIP_BACKEND", "sentence_transformers"),
            model_name=CONFIG.CLIP_MODEL,
            model_id=getattr(CONFIG, "CLIP_MODEL_ID", "sentence-transformers/clip-ViT-B-32"),
            model_revision=CONFIG.CLIP_MODEL_REVISION,
            dimension=self.dimension,
        )

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("texts must contain non-empty strings")

        current_batch_size = batch_size
        while current_batch_size >= 1:
            try:
                res = self.model.encode(
                    list(texts),
                    batch_size=current_batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return _normalize_matrix(res)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and current_batch_size > 1:
                    current_batch_size //= 2
                else:
                    raise
        raise RuntimeError("Failed to encode texts due to OOM")

    def encode_images(
        self,
        images: Sequence[object],
        batch_size: int = CONFIG.batch_size,
    ) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        current_batch_size = batch_size
        while current_batch_size >= 1:
            try:
                res = self.model.encode(
                    list(images),
                    batch_size=current_batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return _normalize_matrix(res)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and current_batch_size > 1:
                    current_batch_size //= 2
                else:
                    raise
        raise RuntimeError("Failed to encode images due to OOM")
