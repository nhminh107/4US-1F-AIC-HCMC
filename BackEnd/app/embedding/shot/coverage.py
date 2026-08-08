"""Coverage-aware weights for overlapping clip windows."""

from __future__ import annotations

from BackEnd.app.contracts.embedding import ClipRecord


def has_overlap(clip_records: list[ClipRecord]) -> bool:
    """Return True when any clip interval overlaps a previous interval."""

    sorted_clips = sorted(clip_records, key=lambda clip: (clip.start_ms, clip.end_ms))
    previous_end: int | None = None
    for clip in sorted_clips:
        if previous_end is not None and clip.start_ms < previous_end:
            return True
        previous_end = max(previous_end or clip.end_ms, clip.end_ms)
    return False


def duration_weights(clip_records: list[ClipRecord]) -> list[float]:
    """Return simple duration weights for non-overlapping clips."""

    return [float(clip.end_ms - clip.start_ms) for clip in clip_records]


def coverage_weights(clip_records: list[ClipRecord]) -> list[float]:
    """Return weights that avoid double-counting overlapping time spans."""

    if not clip_records:
        return []

    boundaries = sorted(
        {
            boundary
            for clip in clip_records
            for boundary in (clip.start_ms, clip.end_ms)
        }
    )
    weights = [0.0 for _ in clip_records]
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        segment_duration = end_ms - start_ms
        if segment_duration <= 0:
            continue
        covering_indexes = [
            index
            for index, clip in enumerate(clip_records)
            if clip.start_ms <= start_ms and clip.end_ms >= end_ms
        ]
        if not covering_indexes:
            continue
        share = segment_duration / len(covering_indexes)
        for index in covering_indexes:
            weights[index] += share
    return weights

