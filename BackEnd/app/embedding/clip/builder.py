"""Build deterministic clip windows from shot metadata."""

from __future__ import annotations

from BackEnd.app.clip_extractor import plan_clip_windows
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

        intervals = plan_clip_windows(
            shot.start_ms,
            shot.end_ms,
            split_threshold_ms=config.window_ms,
            window_ms=config.window_ms,
            stride_ms=config.stride_ms,
            min_new_window_gap_ms=config.min_new_window_gap_ms,
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


def _validate_shot(shot: ShotMetadata) -> None:
    if not shot.shot_id:
        raise ValueError("shot_id must be a non-empty string.")
    if not shot.video_id:
        raise ValueError("video_id must be a non-empty string.")
    if shot.start_ms < 0:
        raise ValueError("shot start_ms must be greater than or equal to 0.")
    if shot.end_ms <= shot.start_ms:
        raise ValueError("shot end_ms must be greater than start_ms.")
