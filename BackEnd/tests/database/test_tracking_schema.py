"""Schema-level tests for independent YOLO tracking observations."""

from BackEnd.app.database.models import ObjectDetection, ObjectTrack, TrackObservation


def test_track_observation_has_no_object_detection_foreign_key() -> None:
    columns = set(TrackObservation.__table__.columns.keys())
    assert columns == {
        "track_id",
        "frame_idx",
        "timestamp_ms",
        "confidence",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }
    assert set(TrackObservation.__table__.primary_key.columns.keys()) == {
        "track_id",
        "frame_idx",
    }
    foreign_keys = {
        foreign_key.target_fullname
        for foreign_key in TrackObservation.__table__.foreign_keys
    }
    assert foreign_keys == {"objecttrack.track_id"}
    assert "track_observation" not in ObjectDetection.__mapper__.relationships


def test_object_track_contains_tracking_provenance() -> None:
    columns = set(ObjectTrack.__table__.columns.keys())
    assert {
        "model_name",
        "model_version",
        "tracker_name",
        "tracker_version",
        "sampling_fps",
        "mapping_version",
    }.issubset(columns)
