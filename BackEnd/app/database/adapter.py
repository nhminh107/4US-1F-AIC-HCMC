"""Convert SQLAlchemy database models into pipeline data contracts."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, TypeVar

from BackEnd.app.contracts.pipeline import (
    CaptionResult,
    ClassMetadata,
    ClipEmbeddingMapping,
    ClipWindowMetadata,
    FrameEmbeddingMapping,
    FrameMetadata,
    ObjectDetectionResult,
    ObjectTrackResult,
    OCRResult,
    ShotEmbeddingMapping,
    ShotMetadata,
    TrackObservationResult,
    TranscriptSegmentResult,
    VideoMetadata,
)
from BackEnd.app.database.models import (
    Caption,
    ClassID,
    ClipEmbeddingRecord,
    ClipWindow,
    Frame,
    FrameEmbeddingRecord,
    ObjectDetection,
    ObjectTrack,
    OCR,
    Shot,
    ShotEmbeddingRecord,
    TrackObservation,
    TranscriptSegment,
    Video,
)

_ContractT = TypeVar("_ContractT")
_FieldConverters = dict[str, Callable[[Any], Any]]


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def _optional_tuple(value: list[str] | None) -> tuple[str, ...] | None:
    return tuple(value) if value is not None else None


def _contract_from_model(
    contract_type: type[_ContractT],
    model: Any,
    converters: _FieldConverters | None = None,
) -> _ContractT:
    """Copy matching model fields into a data contract."""

    field_converters = converters or {}
    values: dict[str, Any] = {}
    for field in fields(contract_type):
        value = getattr(model, field.name)
        converter = field_converters.get(field.name)
        values[field.name] = converter(value) if converter else value
    return contract_type(**values)


def video_metadata_from_video(video: Video) -> VideoMetadata:
    """Convert a database video into pipeline metadata."""

    return _contract_from_model(
        VideoMetadata,
        video,
        {"video_path": Path, "keywords": _optional_tuple},
    )


def shot_metadata_from_shot(shot: Shot) -> ShotMetadata:
    """Convert a database shot into pipeline metadata."""

    return _contract_from_model(ShotMetadata, shot)


def frame_metadata_from_frame(frame: Frame) -> FrameMetadata:
    """Convert a database frame into pipeline metadata."""

    return _contract_from_model(
        FrameMetadata,
        frame,
        {"frame_path": _optional_path},
    )


def class_metadata_from_class_id(class_id: ClassID) -> ClassMetadata:
    """Convert a database class row into pipeline metadata."""

    return _contract_from_model(ClassMetadata, class_id)


def ocr_result_from_ocr(ocr: OCR) -> OCRResult:
    """Convert a database OCR row into a pipeline result."""

    return _contract_from_model(OCRResult, ocr)


def object_detection_result_from_object_detection(
    detection: ObjectDetection,
) -> ObjectDetectionResult:
    """Convert a database detection into a pipeline result."""

    return _contract_from_model(ObjectDetectionResult, detection)


def frame_embedding_mapping_from_record(
    record: FrameEmbeddingRecord,
) -> FrameEmbeddingMapping:
    """Convert a frame embedding record into a pipeline mapping."""

    return _contract_from_model(FrameEmbeddingMapping, record)


def transcript_segment_result_from_segment(
    segment: TranscriptSegment,
) -> TranscriptSegmentResult:
    """Convert a database transcript segment into a pipeline result."""

    return _contract_from_model(TranscriptSegmentResult, segment)


def object_track_result_from_object_track(
    track: ObjectTrack,
) -> ObjectTrackResult:
    """Convert a database object track into a pipeline result."""

    return _contract_from_model(ObjectTrackResult, track)


def track_observation_result_from_observation(
    observation: TrackObservation,
) -> TrackObservationResult:
    """Convert an independent YOLO observation into a pipeline result."""

    return _contract_from_model(TrackObservationResult, observation)


def clip_window_metadata_from_clip_window(
    clip_window: ClipWindow,
) -> ClipWindowMetadata:
    """Convert a database clip window into pipeline metadata."""

    return _contract_from_model(
        ClipWindowMetadata,
        clip_window,
        {"clip_path": _optional_path},
    )


def clip_embedding_mapping_from_record(
    record: ClipEmbeddingRecord,
) -> ClipEmbeddingMapping:
    """Convert a clip embedding record into a pipeline mapping."""

    return _contract_from_model(ClipEmbeddingMapping, record)


def shot_embedding_mapping_from_record(
    record: ShotEmbeddingRecord,
) -> ShotEmbeddingMapping:
    """Convert a shot embedding record into a pipeline mapping."""

    return _contract_from_model(ShotEmbeddingMapping, record)


def caption_result_from_caption(caption: Caption) -> CaptionResult:
    """Convert a database caption into a pipeline result."""

    return _contract_from_model(CaptionResult, caption)


__all__ = [
    "caption_result_from_caption",
    "class_metadata_from_class_id",
    "clip_embedding_mapping_from_record",
    "clip_window_metadata_from_clip_window",
    "frame_embedding_mapping_from_record",
    "frame_metadata_from_frame",
    "object_detection_result_from_object_detection",
    "object_track_result_from_object_track",
    "ocr_result_from_ocr",
    "shot_embedding_mapping_from_record",
    "shot_metadata_from_shot",
    "track_observation_result_from_observation",
    "transcript_segment_result_from_segment",
    "video_metadata_from_video",
]
