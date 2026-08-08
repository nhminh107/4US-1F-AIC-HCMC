"""Quality gates for embedding vectors and batches."""

from __future__ import annotations

import numpy as np

from BackEnd.app.embedding.common.vector import validate_embedding_vector


def validate_embedding_matrix(
    vectors,
    *,
    dimension: int,
    normalized: bool = True,
) -> np.ndarray:
    """Validate a matrix of embedding vectors."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")
    if array.shape[1] != dimension:
        raise ValueError(f"Embedding matrix dimension {array.shape[1]} != expected {dimension}.")
    for vector in array:
        validate_embedding_vector(vector, dimension=dimension, normalized=normalized)
    return array

