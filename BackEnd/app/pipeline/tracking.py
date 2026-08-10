"""Run object tracking for persisted videos and save tracking contracts."""

from __future__ import annotations

from dataclasses import replace

from BackEnd.app.contracts.pipeline import ObjectTrackResult, VideoMetadata
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.tracking.tracking import ByteTrackService


def track_video(
    video: VideoMetadata,
    db: PostgreManager,
    tracker: ByteTrackService,
) -> list[ObjectTrackResult]:
    """Track objects across one video's persisted shots and save all results."""

    frames = db.get_frame_record_by_video_id(video.video_id)
    result = tracker.track_video(
        video,
        db.get_list_shot_in_video(video.video_id),
        frames,
    )
    persisted_frame_ids = {frame.frame_id for frame in frames}
    detections = [
        detection
        for detection in result.detections
        if detection.frame_id in persisted_frame_ids
    ]

    track_id_map: dict[int, int] = {}
    tracks: list[ObjectTrackResult] = []
    for track in result.tracks:
        record = db.add_object_track(
            shot_id=track.shot_id,
            class_id=track.class_id,
            start_ms=track.start_ms,
            end_ms=track.end_ms,
            observation_count=track.observation_count,
            start_frame_idx=track.start_frame_idx,
            end_frame_idx=track.end_frame_idx,
            avg_confidence=track.avg_confidence,
            tracker_name=track.tracker_name,
            tracker_version=track.tracker_version,
        )
        if track.track_id is not None:
            track_id_map[track.track_id] = record.track_id
        tracks.append(replace(track, track_id=record.track_id))

    detection_id_map: dict[int, int] = {}
    for detection in detections:
        record = db.add_object_detection(
            frame_id=detection.frame_id,
            class_id=detection.class_id,
            confidence=detection.confidence,
            x_min=detection.x_min,
            x_max=detection.x_max,
            y_min=detection.y_min,
            y_max=detection.y_max,
            model_name=detection.model_name,
            model_version=detection.model_version,
        )
        if detection.detection_id is not None:
            detection_id_map[detection.detection_id] = record.detection_id

    for observation in result.observations:
        track_id = track_id_map.get(observation.track_id)
        detection_id = detection_id_map.get(observation.detection_id)
        if track_id is not None and detection_id is not None:
            db.add_track_observation(track_id, detection_id)

    return tracks


def track_videos(
    videos: list[VideoMetadata],
    db: PostgreManager,
    tracker: ByteTrackService,
) -> list[ObjectTrackResult]:
    """Track and persist objects for a list of videos with one tracker model."""

    tracks: list[ObjectTrackResult] = []
    for video in videos:
        tracks.extend(track_video(video, db, tracker))
    return tracks


__all__ = ["track_video", "track_videos"]
