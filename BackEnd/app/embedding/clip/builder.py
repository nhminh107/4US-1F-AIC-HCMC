"""Build deterministic clip windows from shot metadata."""

from __future__ import annotations

from BackEnd.app.contracts.embedding import ClipRecord
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding.CONFIG import ClipBuilderConfig
from BackEnd.app.embedding.common.ids import deterministic_id


def build_clips(
    shots: list[ShotMetadata],
    config: ClipBuilderConfig = ClipBuilderConfig(),
) -> list[ClipRecord]:
    """Convert shots into logical clip records without creating media files."""

    clips: list[ClipRecord] = []
    seen_shots: set[str] = set()
    for shot in sorted(shots, key=lambda item: (item.video_id, item.start_ms, item.end_ms)):
        _validate_shot(shot)
        if shot.shot_id in seen_shots:
            continue
        seen_shots.add(shot.shot_id)

        duration_ms = shot.end_ms - shot.start_ms
        intervals = (
            [(shot.start_ms, shot.end_ms, "full_shot")]
            if duration_ms <= config.window_ms
            else _fixed_windows(shot.start_ms, shot.end_ms, config)
        )
        for start_ms, end_ms, scale_type in intervals:
            clips.append(
                ClipRecord(
                    clip_id=deterministic_id(
                        config.dataset_id,
                        shot.video_id,
                        shot.shot_id,
                        start_ms,
                        end_ms,
                        config.clip_builder_version,
                    ),
                    video_id=shot.video_id,
                    shot_id=shot.shot_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    scale_type=scale_type,
                    target_num_frames=config.target_num_frames,
                    sampling_strategy=config.sampling_strategy,
                    sampling_version=config.sampling_version,
                    clip_builder_version=config.clip_builder_version,
                )
            )
    return clips


def _fixed_windows(
    start_ms: int,
    end_ms: int,
    config: ClipBuilderConfig,
) -> list[tuple[int, int, str]]:
    intervals: list[tuple[int, int, str]] = []
    current_start = start_ms
    while current_start + config.window_ms < end_ms:
        intervals.append((current_start, current_start + config.window_ms, "fixed_window"))
        current_start += config.stride_ms

    tail_start = end_ms - config.window_ms
    if not intervals or tail_start - intervals[-1][0] >= config.min_new_window_gap_ms:
        intervals.append((tail_start, end_ms, "fixed_window"))
    else:
        intervals[-1] = (tail_start, end_ms, "fixed_window")

    deduped: list[tuple[int, int, str]] = []
    seen = set()
    for interval in intervals:
        key = (interval[0], interval[1])
        if key not in seen:
            seen.add(key)
            deduped.append(interval)
    return deduped


def _validate_shot(shot: ShotMetadata) -> None:
    if not shot.shot_id:
        raise ValueError("shot_id must be a non-empty string.")
    if not shot.video_id:
        raise ValueError("video_id must be a non-empty string.")
    if shot.start_ms < 0:
        raise ValueError("shot start_ms must be greater than or equal to 0.")
    if shot.end_ms <= shot.start_ms:
        raise ValueError("shot end_ms must be greater than start_ms.")
