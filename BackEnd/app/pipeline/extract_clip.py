"""Create logical clips from persisted shots and store them in PostgreSQL."""

from __future__ import annotations

from pathlib import Path

from BackEnd.app.clip_extractor import ClipExtractor
from BackEnd.app.contracts.pipeline import ClipWindowMetadata
from BackEnd.app.database.postgre_db import PostgreManager


def extract_clip(
    video_id: str,
    db: PostgreManager,
    extractor: ClipExtractor,
) -> list[ClipWindowMetadata]:
    """Create and persist clips for every shot of one video."""

    clips: list[ClipWindowMetadata] = []
    for shot in db.get_list_shot_in_video(video_id):
        for raw_clip in extractor.run(shot):
            clip_path = raw_clip["clip_path"]
            clip = ClipWindowMetadata(
                clip_id=raw_clip["clip_id"],
                shot_id=raw_clip["shot_id"],
                start_ms=raw_clip["start_ms"],
                end_ms=raw_clip["end_ms"],
                start_frame_idx=raw_clip["start_frame_idx"],
                end_frame_idx=raw_clip["end_frame_idx"],
                sampling_fps=raw_clip["sampling_fps"],
                clip_path=Path(clip_path) if clip_path is not None else None,
            )
            db.add_clip(
                clip_id=clip.clip_id,
                shot_id=clip.shot_id,
                start_ms=clip.start_ms,
                end_ms=clip.end_ms,
                start_frame_idx=clip.start_frame_idx,
                end_frame_idx=clip.end_frame_idx,
                sampling_fps=clip.sampling_fps,
                clip_path=str(clip.clip_path) if clip.clip_path is not None else None,
            )
            clips.append(clip)

    return clips


if __name__ == "__main__":
    db = PostgreManager()
    extractor = ClipExtractor()

    extract_clip('L21_V005', db, extractor)
        
