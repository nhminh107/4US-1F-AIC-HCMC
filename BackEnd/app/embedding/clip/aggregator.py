"""Aggregate CLIP frame vectors into clip vectors."""

from __future__ import annotations

import numpy as np

from BackEnd.app.embedding.common.vector import normalize_l2, validate_embedding_vector


def aggregate_clip_frames(frame_vectors, valid_mask) -> np.ndarray:
    """Pool valid frame vectors with masked mean and L2 normalization."""

    vectors = np.asarray(frame_vectors, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if vectors.ndim != 2:
        raise ValueError("frame_vectors must have shape (num_frames, dimension).")
    if mask.ndim != 1 or mask.shape[0] != vectors.shape[0]:
        raise ValueError("valid_mask must match the number of frame vectors.")
    if not mask.any():
        raise ValueError("At least one valid frame vector is required.")

    valid_vectors = vectors[mask]
    for vector in valid_vectors:
        validate_embedding_vector(vector, dimension=vectors.shape[1], normalized=False)
    return normalize_l2(valid_vectors.mean(axis=0, dtype=np.float32))

