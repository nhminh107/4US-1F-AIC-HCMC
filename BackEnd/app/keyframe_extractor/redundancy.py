"""Redundancy elimination for hybrid keyframe candidates."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence, Set

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class RedundancyCandidate:
    """Candidate frame with metadata used for duplicate tie-breaking."""

    frame_idx: int
    image: object
    clip_vector: object
    center_distance: float = 0.0


def eliminate_redundant_candidates(
    candidates: Sequence[RedundancyCandidate],
    *,
    existing_frame_idxs: Sequence[int] | Set[int] | None = None,
    max_output: int,
    hsv_similarity_threshold: float = 0.8,
    clip_similarity_threshold: float = 0.95,
    low_information_min_nonzero_bins: int = 10,
) -> list[int]:
    """Remove low-information and near-duplicate candidate frames."""

    if max_output <= 0:
        raise ValueError("max_output must be positive.")

    existing_set = set(existing_frame_idxs or [])
    kept: list[RedundancyCandidate] = []
    histograms: dict[int, np.ndarray] = {}
    vectors: dict[int, np.ndarray] = {}
    for candidate in candidates:
        histogram = hsv_histogram(candidate.image)
        if np.count_nonzero(histogram) < low_information_min_nonzero_bins:
            continue
        vector = _normalized_vector(candidate.clip_vector)
        kept.append(candidate)
        histograms[candidate.frame_idx] = histogram
        vectors[candidate.frame_idx] = vector

    while True:
        duplicate_pair = _most_similar_duplicate_pair(
            kept,
            histograms,
            vectors,
            hsv_similarity_threshold=hsv_similarity_threshold,
            clip_similarity_threshold=clip_similarity_threshold,
        )
        if duplicate_pair is None:
            break
        remove_idx = _choose_duplicate_to_remove(
            duplicate_pair[0],
            duplicate_pair[1],
            existing_set,
        )
        kept = [candidate for candidate in kept if candidate.frame_idx != remove_idx]

    ranked = sorted(
        kept,
        key=lambda candidate: (
            candidate.center_distance,
            -_distance_to_existing(candidate.frame_idx, existing_set),
            candidate.frame_idx,
        ),
    )
    selected = sorted(candidate.frame_idx for candidate in ranked[:max_output])
    return selected


def hsv_histogram(image: object) -> np.ndarray:
    """Return a normalized 8x8x8 HSV histogram for an RGB image-like object."""

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3).")
        pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")

    hsv = np.asarray(pil_image.convert("HSV"), dtype=np.uint8)
    # Promote before multiplying: uint8 overflow collapsed many histogram bins.
    bins = (hsv // 32).astype(np.int16)
    flat = (bins[:, :, 0] * 64 + bins[:, :, 1] * 8 + bins[:, :, 2]).reshape(-1)
    histogram = np.bincount(flat, minlength=512).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    return histogram if norm == 0 else histogram / norm


def _most_similar_duplicate_pair(
    candidates: Sequence[RedundancyCandidate],
    histograms: Mapping[int, np.ndarray],
    vectors: Mapping[int, np.ndarray],
    *,
    hsv_similarity_threshold: float,
    clip_similarity_threshold: float,
) -> tuple[RedundancyCandidate, RedundancyCandidate] | None:
    best_pair = None
    best_score = -1.0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            hsv_similarity = float(np.dot(histograms[left.frame_idx], histograms[right.frame_idx]))
            clip_similarity = float(np.dot(vectors[left.frame_idx], vectors[right.frame_idx]))
            if (
                hsv_similarity > hsv_similarity_threshold
                and clip_similarity > clip_similarity_threshold
                and clip_similarity > best_score
            ):
                best_pair = (left, right)
                best_score = clip_similarity
    return best_pair


def _choose_duplicate_to_remove(
    left: RedundancyCandidate,
    right: RedundancyCandidate,
    existing_set: set[int],
) -> int:
    left_priority = (
        left.center_distance,
        -_distance_to_existing(left.frame_idx, existing_set),
        left.frame_idx,
    )
    right_priority = (
        right.center_distance,
        -_distance_to_existing(right.frame_idx, existing_set),
        right.frame_idx,
    )
    return right.frame_idx if left_priority <= right_priority else left.frame_idx


def _distance_to_existing(frame_idx: int, existing_set: set[int]) -> int:
    if not existing_set:
        return 1_000_000_000
    return min(abs(frame_idx - existing) for existing in existing_set)


def _normalized_vector(vector: object) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("clip_vector must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError("clip_vector contains NaN or infinity.")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("clip_vector is zero.")
    return array / norm


def eliminate_cross_shot_duplicates(
    per_shot_candidates: Sequence[Sequence[RedundancyCandidate]],
    *,
    hsv_similarity_threshold: float = 0.75,
    clip_similarity_threshold: float = 0.90,
    max_frame_gap: int = 150,
) -> list[list[int]]:
    """Remove cross-shot boundary duplicate candidates between adjacent shots."""

    cleaned_per_shot: list[list[RedundancyCandidate]] = [
        list(shot_cands) for shot_cands in per_shot_candidates
    ]

    for k in range(len(cleaned_per_shot) - 1):
        left_shot = cleaned_per_shot[k]
        right_shot = cleaned_per_shot[k + 1]

        if not left_shot or not right_shot:
            continue

        last_cand = left_shot[-1]
        first_cand = right_shot[0]

        if first_cand.frame_idx - last_cand.frame_idx > max_frame_gap:
            continue

        left_hsv = hsv_histogram(last_cand.image)
        right_hsv = hsv_histogram(first_cand.image)
        hsv_sim = float(np.dot(left_hsv, right_hsv))

        left_clip = _normalized_vector(last_cand.clip_vector)
        right_clip = _normalized_vector(first_cand.clip_vector)
        clip_sim = float(np.dot(left_clip, right_clip))

        if hsv_sim > hsv_similarity_threshold and clip_sim > clip_similarity_threshold:
            # If right shot has multiple keyframes, safely remove the boundary duplicate
            if len(right_shot) > 1:
                cleaned_per_shot[k + 1] = right_shot[1:]
            else:
                # If right shot has only 1 keyframe, keep whichever has smaller center_distance
                if first_cand.center_distance > last_cand.center_distance:
                    cleaned_per_shot[k + 1] = []

    return [[c.frame_idx for c in shot] for shot in cleaned_per_shot]
