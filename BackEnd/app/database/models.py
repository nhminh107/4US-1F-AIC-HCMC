"""SQLAlchemy models for the video analysis PostgreSQL database.

The models in this module mirror ``postgre_script.sql``. PostgreSQL folds the
unquoted table names in that script to lowercase, hence the lowercase
``__tablename__`` values below.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""


class Video(Base):
    """A source video and its descriptive metadata."""

    __tablename__ = "video"
    __table_args__ = (CheckConstraint("duration_ms >= 0"),)

    video_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    video_path: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(String(500))
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String(100)))
    author: Mapped[str | None] = mapped_column(String(50))
    channel_id: Mapped[str | None] = mapped_column(String(50))
    channel_url: Mapped[str | None] = mapped_column(String(500))
    watch_url: Mapped[str | None] = mapped_column(String(500))
    thumbnail_url: Mapped[str | None] = mapped_column(String(500))
    publish_date: Mapped[date | None] = mapped_column(Date)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)

    shots: Mapped[list[Shot]] = relationship(back_populates="video")
    transcript_segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="video"
    )


class Shot(Base):
    """A temporally continuous shot in a video."""

    __tablename__ = "shot"
    __table_args__ = (
        CheckConstraint("shot_index >= 0"),
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
        CheckConstraint("start_frame_idx >= 0"),
        CheckConstraint("end_frame_idx >= start_frame_idx"),
        UniqueConstraint("video_id", "shot_id"),
        UniqueConstraint("video_id", "shot_index"),
    )

    shot_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("video.video_id"), nullable=False
    )
    shot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_frame_idx: Mapped[int | None] = mapped_column(BigInteger)
    end_frame_idx: Mapped[int | None] = mapped_column(BigInteger)

    video: Mapped[Video] = relationship(back_populates="shots")
    frames: Mapped[list[Frame]] = relationship(back_populates="shot")
    object_tracks: Mapped[list[ObjectTrack]] = relationship(back_populates="shot")
    clip_windows: Mapped[list[ClipWindow]] = relationship(back_populates="shot")
    embedding_records: Mapped[list[ShotEmbeddingRecord]] = relationship(
        back_populates="shot"
    )
    captions: Mapped[list[Caption]] = relationship(back_populates="shot")


class Frame(Base):
    """A persisted official or extracted keyframe."""

    __tablename__ = "frame"
    __table_args__ = (
        CheckConstraint("n >= 0"),
        CheckConstraint("pts_time >= 0"),
        CheckConstraint("timestamp_ms >= 0"),
        CheckConstraint("fps > 0"),
        CheckConstraint("frame_idx >= 0"),
        CheckConstraint("source IN ('official', 'extracted')"),
        CheckConstraint(
            "source <> 'official' OR shot_id IS NULL",
            name="frame_official_shot_null_check",
        ),
        CheckConstraint("width > 0"),
        CheckConstraint("height > 0"),
        Index(
            "uq_frame_official_video_n",
            "video_id",
            "n",
            unique=True,
            postgresql_where=text("source = 'official'"),
        ),
        ForeignKeyConstraint(
            ["video_id", "shot_id"],
            ["shot.video_id", "shot.shot_id"],
        ),
    )

    frame_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    n: Mapped[int | None] = mapped_column(Integer)
    video_id: Mapped[str] = mapped_column(String(15), nullable=False)
    shot_id: Mapped[str | None] = mapped_column(String(15))
    pts_time: Mapped[float | None] = mapped_column(Float)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    frame_idx: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    frame_role: Mapped[str] = mapped_column(String(20), default="keyframe", server_default="keyframe")
    frame_path: Mapped[str | None] = mapped_column(String(200))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)

    shot: Mapped[Shot | None] = relationship(back_populates="frames")
    ocr_records: Mapped[list[OCR]] = relationship(back_populates="frame")
    object_detections: Mapped[list[ObjectDetection]] = relationship(
        back_populates="frame"
    )
    embedding_records: Mapped[list[FrameEmbeddingRecord]] = relationship(
        back_populates="frame"
    )
    captions: Mapped[list[Caption]] = relationship(back_populates="frame")


class ClassID(Base):
    """An object-detection class label."""

    __tablename__ = "classid"

    class_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    class_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)

    object_detections: Mapped[list[ObjectDetection]] = relationship(
        back_populates="object_class"
    )
    object_tracks: Mapped[list[ObjectTrack]] = relationship(
        back_populates="object_class"
    )


class OCR(Base):
    """Recognized text and its normalized bounding box in a frame."""

    __tablename__ = "ocr"
    __table_args__ = (
        CheckConstraint("n >= 0"),
        CheckConstraint("x_min BETWEEN 0 AND 1"),
        CheckConstraint("x_max BETWEEN 0 AND 1"),
        CheckConstraint("y_min BETWEEN 0 AND 1"),
        CheckConstraint("y_max BETWEEN 0 AND 1"),
        CheckConstraint("x_min < x_max"),
        CheckConstraint("y_min < y_max"),
    )

    frame_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("frame.frame_id"), primary_key=True
    )
    n: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(10))
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)

    frame: Mapped[Frame] = relationship(back_populates="ocr_records")


class ObjectDetection(Base):
    """An object detected in a frame."""

    __tablename__ = "objectdetection"
    __table_args__ = (
        CheckConstraint("confidence BETWEEN 0 AND 1"),
        CheckConstraint("x_min BETWEEN 0 AND 1"),
        CheckConstraint("x_max BETWEEN 0 AND 1"),
        CheckConstraint("y_min BETWEEN 0 AND 1"),
        CheckConstraint("y_max BETWEEN 0 AND 1"),
        CheckConstraint("x_min < x_max"),
        CheckConstraint("y_min < y_max"),
    )

    detection_id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    frame_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("frame.frame_id"), nullable=False
    )
    class_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("classid.class_id"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(50))

    frame: Mapped[Frame] = relationship(back_populates="object_detections")
    object_class: Mapped[ClassID] = relationship(back_populates="object_detections")


class FrameEmbeddingRecord(Base):
    """A mapping from a frame to an entry in a FAISS index."""

    __tablename__ = "frameembeddingrecord"
    __table_args__ = (UniqueConstraint("frame_id", "index_version"),)

    faiss_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    index_version: Mapped[int] = mapped_column(
        Integer, CheckConstraint("index_version >= 0"), primary_key=True
    )
    frame_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("frame.frame_id"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    frame: Mapped[Frame] = relationship(back_populates="embedding_records")


class TranscriptSegment(Base):
    """An ASR transcript segment within a video time range."""

    __tablename__ = "transcriptsegment"
    __table_args__ = (
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
    )

    segment_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("video.video_id"), nullable=False
    )
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(10))

    video: Mapped[Video] = relationship(back_populates="transcript_segments")


class ObjectTrack(Base):
    """A tracked object spanning multiple detections in one shot."""

    __tablename__ = "objecttrack"
    __table_args__ = (
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
        CheckConstraint("start_frame_idx >= 0"),
        CheckConstraint("end_frame_idx >= start_frame_idx"),
        CheckConstraint("observation_count > 0"),
        CheckConstraint("avg_confidence BETWEEN 0 AND 1"),
        CheckConstraint("sampling_fps > 0"),
    )

    track_id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    shot_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("shot.shot_id"), nullable=False
    )
    class_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("classid.class_id"), nullable=False
    )
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_frame_idx: Mapped[int | None] = mapped_column(BigInteger)
    end_frame_idx: Mapped[int | None] = mapped_column(BigInteger)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_confidence: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    tracker_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tracker_version: Mapped[str] = mapped_column(String(50), nullable=False)
    sampling_fps: Mapped[float] = mapped_column(Float, nullable=False)
    mapping_version: Mapped[str] = mapped_column(String(50), nullable=False)

    shot: Mapped[Shot] = relationship(back_populates="object_tracks")
    object_class: Mapped[ClassID] = relationship(back_populates="object_tracks")
    observations: Mapped[list[TrackObservation]] = relationship(
        back_populates="track"
    )


class TrackObservation(Base):
    """One normalized YOLO observation belonging to an object track."""

    __tablename__ = "trackobservation"
    __table_args__ = (
        CheckConstraint("frame_idx >= 0"),
        CheckConstraint("timestamp_ms >= 0"),
        CheckConstraint("confidence BETWEEN 0 AND 1"),
        CheckConstraint("x_min BETWEEN 0 AND 1"),
        CheckConstraint("x_max BETWEEN 0 AND 1"),
        CheckConstraint("y_min BETWEEN 0 AND 1"),
        CheckConstraint("y_max BETWEEN 0 AND 1"),
        CheckConstraint("x_min < x_max"),
        CheckConstraint("y_min < y_max"),
    )

    track_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("objecttrack.track_id"), primary_key=True
    )
    frame_idx: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)

    track: Mapped[ObjectTrack] = relationship(back_populates="observations")


class ClipWindow(Base):
    """A clip sampled from a time window within one shot."""

    __tablename__ = "clipwindow"
    __table_args__ = (
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
        CheckConstraint("start_frame_idx >= 0"),
        CheckConstraint("end_frame_idx >= start_frame_idx"),
        CheckConstraint("sampling_fps > 0"),
        UniqueConstraint("shot_id", "start_ms", "end_ms"),
    )

    clip_id: Mapped[str] = mapped_column(String(15), primary_key=True)
    shot_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("shot.shot_id"), nullable=False
    )
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_frame_idx: Mapped[int | None] = mapped_column(BigInteger)
    end_frame_idx: Mapped[int | None] = mapped_column(BigInteger)
    sampling_fps: Mapped[float | None] = mapped_column(Float)
    clip_path: Mapped[str | None] = mapped_column(String(200))

    shot: Mapped[Shot] = relationship(back_populates="clip_windows")
    embedding_records: Mapped[list[ClipEmbeddingRecord]] = relationship(
        back_populates="clip"
    )
    captions: Mapped[list[Caption]] = relationship(back_populates="clip")


class ClipEmbeddingRecord(Base):
    """A mapping from a clip window to an entry in a FAISS index."""

    __tablename__ = "clipembeddingrecord"
    __table_args__ = (UniqueConstraint("clip_id", "index_version"),)

    faiss_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    index_version: Mapped[int] = mapped_column(
        Integer, CheckConstraint("index_version >= 0"), primary_key=True
    )
    clip_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("clipwindow.clip_id"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    clip: Mapped[ClipWindow] = relationship(back_populates="embedding_records")


class ShotEmbeddingRecord(Base):
    """A mapping from a pooled shot embedding to a FAISS entry."""

    __tablename__ = "shotembeddingrecord"
    __table_args__ = (
        UniqueConstraint(
            "shot_id",
            "index_version",
            "model_name",
            "model_version",
            "pooling_method",
        ),
    )

    faiss_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    index_version: Mapped[int] = mapped_column(
        Integer, CheckConstraint("index_version >= 0"), primary_key=True
    )
    shot_id: Mapped[str] = mapped_column(
        String(15), ForeignKey("shot.shot_id"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    pooling_method: Mapped[str] = mapped_column(
        String(30), nullable=False, default="mean", server_default="mean"
    )

    shot: Mapped[Shot] = relationship(back_populates="embedding_records")


class Caption(Base):
    """A generated caption attached to exactly one frame, clip, or shot."""

    __tablename__ = "caption"
    __table_args__ = (
        CheckConstraint("num_nonnulls(frame_id, clip_id, shot_id) = 1"),
        CheckConstraint("confidence BETWEEN 0 AND 1"),
    )

    caption_id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    frame_id: Mapped[str | None] = mapped_column(
        String(15), ForeignKey("frame.frame_id")
    )
    clip_id: Mapped[str | None] = mapped_column(
        String(15), ForeignKey("clipwindow.clip_id")
    )
    shot_id: Mapped[str | None] = mapped_column(
        String(15), ForeignKey("shot.shot_id")
    )
    caption_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    frame: Mapped[Frame | None] = relationship(back_populates="captions")
    clip: Mapped[ClipWindow | None] = relationship(back_populates="captions")
    shot: Mapped[Shot | None] = relationship(back_populates="captions")
