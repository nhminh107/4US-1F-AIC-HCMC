"""Pipeline adapter for idempotent additional-keyframe extraction."""

from __future__ import annotations

from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.keyframe_extractor import KeyframeExtractor


def extract_keyframes(video_id: str, db: PostgreManager, extractor: KeyframeExtractor) -> None:
    """Extract only frame indices absent from the database.

    This adapter deliberately preserves the existing pipeline I/O contract:
    only extracted JPEGs and their Frame database records are emitted.
    """

    existing_frames = db.get_frame_record_by_video_id(video_id)
    keyframes = extractor.extract_for_video(
        video_id=video_id,
        shots=db.get_list_shot_in_video(video_id),
        existing_frame_idxs=[frame.frame_idx for frame in existing_frames],
        existing_frame_ids=[frame.frame_id for frame in existing_frames],
    )

    for kf in keyframes:
        db.add_frame(
            frame_id=kf.frame_id, 
            video_id=video_id, 
            shot_id=kf.shot_id,
            timestamp_ms=kf.timestamp_ms,
            fps = kf.fps, 
            frame_idx=kf.frame_idx,
            source=kf.source,
            n = kf.n, 
            pts_time=kf.pts_time, 
            frame_path=str(kf.frame_path),
            width=kf.width,
            height=kf.height
        )

if __name__ == "__main__":
    database = PostgreManager()
    keyframe_extractor = KeyframeExtractor()
    extract_keyframes("L23_V005", database, keyframe_extractor)
