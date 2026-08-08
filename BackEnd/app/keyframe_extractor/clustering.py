"""Deterministic clustering helpers for hybrid keyframe selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterSelection:
    """Representative frame chosen from one feature cluster."""

    frame_idx: int
    vector_row: int
    center_distance: float


def select_cluster_representatives(
    frame_indices: Sequence[int],
    vectors,
    *,
    max_representatives: int,
) -> list[ClusterSelection]:
    """Select frame indexes nearest to deterministic KMeans centers."""

    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    matrix = np.asarray(vectors, dtype=np.float32)
    if len(frame_indices) == 0:
        return []
    if matrix.ndim != 2 or matrix.shape[0] != len(frame_indices):
        raise ValueError("vectors must be a 2-D matrix with one row per frame index.")
    if max_representatives <= 0:
        raise ValueError("max_representatives must be positive.")
    if len(frame_indices) == 1:
        return [ClusterSelection(frame_indices[0], 0, 0.0)]

    max_k = min(max_representatives, len(frame_indices), max(1, math.ceil(math.sqrt(len(frame_indices)))))
    if max_k == 1:
        row, distance = _nearest_to_center(matrix, np.mean(matrix, axis=0))
        return [ClusterSelection(frame_indices[row], row, distance)]

    best_labels = None
    best_centers = None
    best_score = -float("inf")
    for k in range(2, max_k + 1):
        labels, centers = _kmeans(matrix, k)
        if len(set(labels.tolist())) < 2:
            continue
        score = _silhouette_score(matrix, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_centers = centers

    if best_labels is None or best_centers is None:
        row, distance = _nearest_to_center(matrix, np.mean(matrix, axis=0))
        return [ClusterSelection(frame_indices[row], row, distance)]

    selections: list[ClusterSelection] = []
    for cluster_id in sorted(set(best_labels.tolist())):
        member_rows = np.flatnonzero(best_labels == cluster_id)
        center = best_centers[cluster_id]
        row, distance = _nearest_member_to_center(matrix, member_rows, center)
        selections.append(ClusterSelection(frame_indices[row], int(row), distance))

    return sorted(selections, key=lambda item: item.frame_idx)[:max_representatives]


def _kmeans(matrix: np.ndarray, k: int, *, max_iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    centers = _initial_centers(matrix, k)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(max_iterations):
        distances = _pairwise_distances(matrix, centers)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = matrix[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = np.mean(members, axis=0)
    return labels, centers


def _initial_centers(matrix: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.mean(matrix, axis=0, keepdims=True)
    order = np.argsort(matrix[:, 0], kind="mergesort")
    positions = [round(i * (len(order) - 1) / (k - 1)) for i in range(k)]
    return np.asarray([matrix[order[position]] for position in positions], dtype=np.float32)


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _nearest_to_center(matrix: np.ndarray, center: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(matrix - center, axis=1)
    row = int(np.argmin(distances))
    return row, float(distances[row])


def _nearest_member_to_center(
    matrix: np.ndarray, member_rows: np.ndarray, center: np.ndarray
) -> tuple[int, float]:
    distances = np.linalg.norm(matrix[member_rows] - center, axis=1)
    offset = int(np.argmin(distances))
    row = int(member_rows[offset])
    return row, float(distances[offset])


def _silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return -float("inf")
    distances = _pairwise_distances(matrix, matrix)
    scores: list[float] = []
    for row in range(len(matrix)):
        same_cluster = labels == labels[row]
        same_cluster[row] = False
        if np.any(same_cluster):
            a = float(np.mean(distances[row, same_cluster]))
        else:
            a = 0.0

        b_values = [
            float(np.mean(distances[row, labels == other_label]))
            for other_label in unique_labels
            if other_label != labels[row]
        ]
        b = min(b_values) if b_values else 0.0
        denominator = max(a, b)
        scores.append(0.0 if denominator == 0 else (b - a) / denominator)
    return float(np.mean(scores))
