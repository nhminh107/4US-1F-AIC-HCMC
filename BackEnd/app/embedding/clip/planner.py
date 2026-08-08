"""Plan per-video clip embedding work."""

from __future__ import annotations

from collections import defaultdict

from BackEnd.app.contracts.embedding import ClipRecord, VideoAsset, VideoWorkUnit
from BackEnd.app.embedding.clip.sampler import uniform_midpoint_timestamps


def plan_video_work(
    clips: list[ClipRecord],
    video_assets: dict[str, VideoAsset] | None = None,
    max_clips_per_unit: int | None = None,
) -> list[VideoWorkUnit]:
    """Group clips by video and deduplicate sampled timestamps."""

    grouped: dict[str, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        grouped[clip.video_id].append(clip)

    work_units: list[VideoWorkUnit] = []
    for video_id in sorted(grouped):
        sorted_clips = tuple(sorted(grouped[video_id], key=lambda item: (item.start_ms, item.end_ms, item.clip_id)))
        
        # Split sorted_clips into chunks if max_clips_per_unit is set
        if max_clips_per_unit is not None and max_clips_per_unit > 0:
            clip_chunks = [
                sorted_clips[i : i + max_clips_per_unit]
                for i in range(0, len(sorted_clips), max_clips_per_unit)
            ]
        else:
            clip_chunks = [sorted_clips]

        for chunk in clip_chunks:
            requested: list[int] = []
            timestamp_to_clip_ids: dict[int, list[str]] = defaultdict(list)
            for clip in chunk:
                for timestamp in uniform_midpoint_timestamps(clip):
                    requested.append(timestamp)
                    timestamp_to_clip_ids[timestamp].append(clip.clip_id)

            unique_timestamps = tuple(sorted(set(requested)))
            work_units.append(
                VideoWorkUnit(
                    video_id=video_id,
                    video_asset=None if video_assets is None else video_assets.get(video_id),
                    sorted_clip_records=tuple(chunk),
                    requested_timestamps_ms=tuple(requested),
                    unique_timestamps_ms=unique_timestamps,
                    timestamp_to_clip_ids={
                        timestamp: tuple(clip_ids)
                        for timestamp, clip_ids in timestamp_to_clip_ids.items()
                    },
                )
            )
    return work_units


def dedup_ratio(work_unit: VideoWorkUnit) -> float:
    """Return fraction of requested timestamps eliminated by deduplication."""

    requested_count = len(work_unit.requested_timestamps_ms)
    if requested_count == 0:
        return 0.0
    return 1.0 - (len(work_unit.unique_timestamps_ms) / requested_count)

