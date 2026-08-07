"""Vector normalization and validation utilities."""

from __future__ import annotations

import numpy as np


def normalize_l2(vector, *, epsilon: float = 1e-12) -> np.ndarray:
    """Return a float32 L2-normalized vector."""

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= epsilon:
        raise ValueError("Embedding vector norm is invalid or zero.")
    return array / norm


def validate_embedding_vector(
    vector,
    *,
    dimension: int | None = None,
    normalized: bool = True,
    norm_tolerance: float = 1e-3,
) -> np.ndarray:
    """Validate vector shape, finite values, and optional L2 normalization."""

    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("Embedding vector must be one-dimensional.")
    if dimension is not None and array.shape[0] != dimension:
        raise ValueError(f"Embedding dimension {array.shape[0]} != expected {dimension}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding vector contains NaN or infinity.")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("Embedding vector is zero.")
    if normalized and abs(norm - 1.0) > norm_tolerance:
        raise ValueError(f"Embedding vector norm {norm:.6f} is not close to 1.")
    return array

