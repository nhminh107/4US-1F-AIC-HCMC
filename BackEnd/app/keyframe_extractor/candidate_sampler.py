"""Sparse candidate frame sampling for hybrid keyframe extraction."""

from __future__ import annotations

from collections.abc import Sequence, Set


def sample_candidate_frame_indices(
    *,
    start_frame_idx: int,
    end_frame_idx: int,
    fps: float,
    existing_frame_idxs: Sequence[int] | Set[int] | None = None,
    sample_fps: float = 1.0,
    max_candidate_frames_per_shot: int = 64,
    min_frame_gap: int = 5,
    transition_margin_frames: int = 2,
) -> list[int]:
    """Return sparse candidate frame indexes inside one shot.

    Candidates are intended for CLIP scoring, not final JPEG output. They stay
    within shot boundaries, avoid already-known keyframes, and are capped so a
    long shot cannot create an unbounded model batch.
    """

    if start_frame_idx < 0:
        raise ValueError("start_frame_idx must be non-negative.")
    if end_frame_idx < start_frame_idx:
        raise ValueError("end_frame_idx must be >= start_frame_idx.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    if max_candidate_frames_per_shot <= 0:
        raise ValueError("max_candidate_frames_per_shot must be positive.")
    if min_frame_gap < 0:
        raise ValueError("min_frame_gap must be non-negative.")
    if transition_margin_frames < 0:
        raise ValueError("transition_margin_frames must be non-negative.")

    existing_set = set(existing_frame_idxs or [])
    inner_start = start_frame_idx
    inner_end = end_frame_idx
    if end_frame_idx - start_frame_idx + 1 > (2 * transition_margin_frames + 1):
        inner_start += transition_margin_frames
        inner_end -= transition_margin_frames

    if inner_end < inner_start:
        inner_start, inner_end = start_frame_idx, end_frame_idx

    step_frames = max(1, round(fps / sample_fps))
    candidates = list(range(inner_start, inner_end + 1, step_frames))
    center = (inner_start + inner_end) // 2
    if center not in candidates:
        candidates.append(center)

    filtered = [
        frame_idx
        for frame_idx in sorted(set(candidates))
        if _is_far_enough(frame_idx, existing_set, min_frame_gap)
    ]

    if not filtered:
        fallback = _nearest_allowed(
            center,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
            existing_set=existing_set,
            min_frame_gap=min_frame_gap,
        )
        return [fallback] if fallback is not None else []

    if len(filtered) <= max_candidate_frames_per_shot:
        return filtered

    return _evenly_pick(filtered, max_candidate_frames_per_shot)


def _is_far_enough(frame_idx: int, existing_set: set[int], min_frame_gap: int) -> bool:
    return all(abs(frame_idx - existing) > min_frame_gap for existing in existing_set)


def _nearest_allowed(
    target: int,
    *,
    start_frame_idx: int,
    end_frame_idx: int,
    existing_set: set[int],
    min_frame_gap: int,
) -> int | None:
    if _is_far_enough(target, existing_set, min_frame_gap):
        return target
    for offset in range(1, end_frame_idx - start_frame_idx + 1):
        for candidate in (target - offset, target + offset):
            if (
                start_frame_idx <= candidate <= end_frame_idx
                and _is_far_enough(candidate, existing_set, min_frame_gap)
            ):
                return candidate
    return None


def _evenly_pick(values: list[int], count: int) -> list[int]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indexes = [
        round(i * (len(values) - 1) / (count - 1))
        for i in range(count)
    ]
    return [values[index] for index in sorted(set(indexes))]
