"""Exact matrix retrieval for embedding artifact benchmarks."""

from __future__ import annotations

import numpy as np


def exact_top_k(query_vectors, visual_vectors, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine/inner-product Top-K scores and indexes without FAISS."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    queries = _as_matrix(query_vectors, "query_vectors")
    visuals = _as_matrix(visual_vectors, "visual_vectors")
    if queries.shape[1] != visuals.shape[1]:
        raise ValueError("Query and visual vectors must have the same dimension.")
    if visuals.shape[0] == 0:
        return (
            np.empty((queries.shape[0], 0), dtype=np.float32),
            np.empty((queries.shape[0], 0), dtype=np.int64),
        )
    similarities = queries @ visuals.T
    result_count = min(top_k, visuals.shape[0])
    indexes = np.argsort(-similarities, axis=1)[:, :result_count]
    scores = np.take_along_axis(similarities, indexes, axis=1)
    return scores.astype(np.float32), indexes.astype(np.int64)


def recall_at_k(top_k_indexes, relevant_indexes: list[set[int]], k: int) -> float:
    """Compute query-level Recall@K for labeled exact retrieval results."""

    indexes = np.asarray(top_k_indexes)
    if indexes.shape[0] != len(relevant_indexes):
        raise ValueError("Number of query result rows must match relevant labels.")
    if not relevant_indexes:
        return 0.0
    hits = 0
    for row, relevant in zip(indexes[:, :k], relevant_indexes):
        hits += int(bool(set(row.tolist()) & set(relevant)))
    return hits / len(relevant_indexes)


def mean_reciprocal_rank(top_k_indexes, relevant_indexes: list[set[int]]) -> float:
    """Compute MRR for labeled exact retrieval results."""

    indexes = np.asarray(top_k_indexes)
    if indexes.shape[0] != len(relevant_indexes):
        raise ValueError("Number of query result rows must match relevant labels.")
    reciprocal_ranks = []
    for row, relevant in zip(indexes, relevant_indexes):
        rank = 0
        for index, value in enumerate(row.tolist(), start=1):
            if value in relevant:
                rank = index
                break
        reciprocal_ranks.append(0.0 if rank == 0 else 1.0 / rank)
    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def _as_matrix(values, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{field_name} must be a vector or matrix.")
    return array

