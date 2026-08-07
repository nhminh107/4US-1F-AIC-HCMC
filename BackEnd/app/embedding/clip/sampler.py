"""Timestamp samplers for clip embedding."""

from __future__ import annotations

from BackEnd.app.contracts.embedding import ClipRecord


def uniform_midpoint_timestamps(clip: ClipRecord, target_num_frames: int | None = None) -> tuple[int, ...]:
    """Return deterministic midpoint timestamps inside a half-open clip interval."""

    frame_count = target_num_frames if target_num_frames is not None else clip.target_num_frames
    if frame_count <= 0:
        raise ValueError("target_num_frames must be greater than 0.")

    duration = clip.end_ms - clip.start_ms
    if duration <= 0:
        raise ValueError("clip duration must be positive.")

    timestamps = []
    for index in range(frame_count):
        timestamp = clip.start_ms + int(((index + 0.5) * duration) / frame_count)
        timestamps.append(min(max(timestamp, clip.start_ms), clip.end_ms - 1))
    return tuple(timestamps)

