"""Aggregate clip vectors into shot vectors."""

from __future__ import annotations

import numpy as np

from BackEnd.app.contracts.embedding import ClipRecord
from BackEnd.app.embedding.common.vector import normalize_l2, validate_embedding_vector
from BackEnd.app.embedding.shot.coverage import coverage_weights, duration_weights, has_overlap


def aggregate_shot_clips(clip_records: list[ClipRecord], clip_vectors) -> np.ndarray:
    """Aggregate compatible clip vectors with duration or coverage-aware weights."""

    vectors = np.asarray(clip_vectors, dtype=np.float32)
    if len(clip_records) == 0:
        raise ValueError("At least one clip is required to aggregate a shot vector.")
    if vectors.ndim != 2 or vectors.shape[0] != len(clip_records):
        raise ValueError("clip_vectors must have one row per clip record.")

    for vector in vectors:
        validate_embedding_vector(vector, dimension=vectors.shape[1], normalized=False)

    if len(clip_records) == 1:
        return normalize_l2(vectors[0])

    weights = coverage_weights(clip_records) if has_overlap(clip_records) else duration_weights(clip_records)
    weight_array = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weight_array).all() or float(weight_array.sum()) <= 0:
        raise ValueError("Shot aggregation weights are invalid.")
    pooled = np.average(vectors, axis=0, weights=weight_array)
    return normalize_l2(pooled)

