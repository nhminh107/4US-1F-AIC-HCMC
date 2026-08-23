"""Deterministic facility-location selection for CLIP keyframe candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_facility_representatives(
    frame_indices: Sequence[int],
    vectors: object,
    *,
    max_representatives: int,
    reference_vectors: object | None = None,
    min_marginal_gain: float = 0.01,
) -> list[int]:
    """Greedily select frames that add the most semantic coverage.

    Existing official/extracted frames can be provided as ``reference_vectors``.
    They seed coverage, so visually equivalent candidates add little or no gain.
    Input vectors are normalized internally and the result is deterministic.
    """

    if max_representatives <= 0:
        raise ValueError("max_representatives must be positive.")
    if min_marginal_gain < 0:
        raise ValueError("min_marginal_gain must be non-negative.")

    matrix = _normalise_matrix(vectors, expected_rows=len(frame_indices))
    if not frame_indices:
        return []

    similarity = np.clip(matrix @ matrix.T, 0.0, 1.0)
    if reference_vectors is None:
        covered = np.zeros(len(frame_indices), dtype=np.float32)
    else:
        references = _normalise_matrix(reference_vectors)
        covered = (
            np.clip(matrix @ references.T, 0.0, 1.0).max(axis=1)
            if len(references)
            else np.zeros(len(frame_indices), dtype=np.float32)
        )

    selected_rows: list[int] = []
    available = set(range(len(frame_indices)))
    for _ in range(min(max_representatives, len(frame_indices))):
        best_row = max(
            available,
            key=lambda row: (
                float(np.maximum(covered, similarity[:, row]).sum() - covered.sum()),
                -frame_indices[row],
            ),
        )
        gain = float(np.maximum(covered, similarity[:, best_row]).mean() - covered.mean())
        if gain < min_marginal_gain and (selected_rows or reference_vectors is not None):
            break
        selected_rows.append(best_row)
        available.remove(best_row)
        covered = np.maximum(covered, similarity[:, best_row])

    return sorted(frame_indices[row] for row in selected_rows)


def _normalise_matrix(vectors: object, *, expected_rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must have shape (count, embedding_dimension).")
    if expected_rows is not None and len(matrix) != expected_rows:
        raise ValueError("frame_indices and vectors must have the same length.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain NaN or infinity.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("vectors contain a zero vector.")
    return matrix / norms
