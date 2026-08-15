"""Run and persist object tracking for one video."""

from __future__ import annotations

from BackEnd.app.contracts.pipeline import ObjectTrackResult, VideoMetadata
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.tracking.tracking import YOLOTrackingService


def track_video(
    video: VideoMetadata,
    db: PostgreManager,
    tracker: YOLOTrackingService,
) -> list[ObjectTrackResult]:
    """Track one video's shots and persist its tracks and observations."""

    shots = db.get_list_shot_in_video(video.video_id)
    result = tracker.track_video(video, shots)
    tracks = db.add_tracking_result(
        tracks=result.tracks,
        observations=result.observations,
    )
    return tracks


__all__ = ["track_video"]

if __name__ == "__main__": 
    db = PostgreManager()
    tracker = YOLOTrackingService()

    videos = db.get_list_video()
    for video in videos: 
        if video.video_id == "L23_V005": 
            track_video(video, db, tracker)
            break