"""SQLAlchemy models for the PostgreSQL video metadata database."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Double,
    ForeignKey,
    Index,
    PrimaryKeyConstraint,
    REAL,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""


class Video(Base):
    __tablename__ = "video"
    __table_args__ = (
        CheckConstraint("fps > 0"), 
        CheckConstraint("duration_ms >= 0")
    )

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    fps: Mapped[float] = mapped_column(Double, nullable=False) 
    duration_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    video_path: Mapped[str] = mapped_column(Text, nullable=False) 

    scenes: Mapped[list[Scene]] = relationship(
        back_populates="video", 
        cascade=("all, delete-orphan"),
        passive_deletes=True
    )

    transcript_segments: Mapped[list[TranscriptSegment]] = relationship(
            back_populates="video",
            cascade="all, delete-orphan",
            passive_deletes=True,
    )

class Scene(Base):
    """A time range belonging to a video."""

    __tablename__ = "scene"
    __table_args__ = (
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
        Index("idx_scene_video", "video_id"),
        Index("idx_scene_video_time", "video_id", "start_ms", "end_ms"),
    )

    scene_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("video.video_id", name="fk_scene_video", ondelete="CASCADE"),
        nullable=False,
    )
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    video: Mapped[Video] = relationship(back_populates="scenes")
    keyframes: Mapped[list[Keyframe]] = relationship(
        back_populates="scene",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Keyframe(Base):
    """A representative frame extracted from a scene."""

    __tablename__ = "keyframe"
    __table_args__ = (
        CheckConstraint("timestamp_ms >= 0"),
        UniqueConstraint(
            "scene_id",
            "timestamp_ms",
            name="uq_keyframe_scene_time",
        ),
        Index("idx_keyframe_scene", "scene_id"),
        Index("idx_keyframe_scene_time", "scene_id", "timestamp_ms"),
    )

    keyframe_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(
        String(48),
        ForeignKey("scene.scene_id", name="fk_keyframe_scene", ondelete="CASCADE"),
        nullable=False,
    )
    visual_index_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        unique=True,
    )
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)

    scene: Mapped[Scene] = relationship(back_populates="keyframes")
    ocr_records: Mapped[list[OCR]] = relationship(
        back_populates="keyframe",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    captions: Mapped[list[Caption]] = relationship(
        back_populates="keyframe",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    embedding_records: Mapped[list[EmbeddingRecord]] = relationship(
        back_populates="keyframe",
        passive_deletes="all",
    )


class OCR(Base):
    """A normalized OCR bounding box and its recognized text."""

    __tablename__ = "ocr"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0 AND 1",
        ),
        CheckConstraint(
            "bbox_x >= 0 "
            "AND bbox_y >= 0 "
            "AND bbox_width > 0 "
            "AND bbox_height > 0 "
            "AND bbox_x + bbox_width <= 1 "
            "AND bbox_y + bbox_height <= 1",
            name="chk_ocr_bbox",
        ),
        Index("idx_ocr_keyframe", "keyframe_id"),
    )

    ocr_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    keyframe_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "keyframe.keyframe_id",
            name="fk_ocr_keyframe",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL, nullable=True)
    bbox_x: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_width: Mapped[float] = mapped_column(Double, nullable=False)
    bbox_height: Mapped[float] = mapped_column(Double, nullable=False)

    keyframe: Mapped[Keyframe] = relationship(back_populates="ocr_records")


class Caption(Base):
    """A caption associated with a keyframe."""

    __tablename__ = "caption"

    caption_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    keyframe_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "keyframe.keyframe_id",
            name="fk_caption_keyframe",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    keyframe: Mapped[Keyframe] = relationship(back_populates="captions")


class TranscriptSegment(Base):
    """An ASR transcript segment within a video time range."""

    __tablename__ = "transcript_segment"
    __table_args__ = (
        CheckConstraint("start_ms >= 0"),
        CheckConstraint("end_ms > start_ms"),
        Index(
            "idx_transcript_video_time",
            "video_id",
            "start_ms",
            "end_ms",
        ),
    )

    segment_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey(
            "video.video_id",
            name="fk_transcript_video",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    video: Mapped[Video] = relationship(back_populates="transcript_segments")


class EmbeddingRecord(Base):
    """Metadata mapping a FAISS entry to its source keyframe."""

    __tablename__ = "embedding_record"
    __table_args__ = (
        PrimaryKeyConstraint("faiss_id", "index_version"),
    )

    faiss_id: Mapped[int] = mapped_column(BigInteger)
    keyframe_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("keyframe.keyframe_id"),
        nullable=True,
    )
    index_version: Mapped[str] = mapped_column(String(50))
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    keyframe: Mapped[Keyframe | None] = relationship(
        back_populates="embedding_records",
    )
