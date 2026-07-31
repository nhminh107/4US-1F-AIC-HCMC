"""PostgreSQL connection and basic database operations."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from BackEnd.app.database.models import (
    Base,
    Caption,
    EmbeddingRecord,
    Keyframe,
    OCR,
    Scene,
    TranscriptSegment,
    Video,
)

load_dotenv()


class Postgre_Manager:
    """Manage PostgreSQL sessions and database records."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        echo: bool = False,
    ) -> None:
        resolved_url = database_url or os.getenv("DATABASE_URL")
        if not resolved_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. "
                "Set it in the environment or pass database_url explicitly."
            )

        self.engine: Engine = create_engine(
            resolved_url,
            echo=echo,
            pool_pre_ping=True,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def init_db(self) -> None:
        """Create tables that do not already exist."""

        Base.metadata.create_all(bind=self.engine)

    def add_video(
        self,
        video_id: str,
        fps: float,
        duration_ms: int,
        video_path: str,
    ) -> Video:
        """Insert a video record and return the persisted ORM object."""

        video = Video(
            video_id=video_id,
            fps=fps,
            duration_ms=duration_ms,
            video_path=video_path,
        )

        with self.session_factory() as session:
            try:
                session.add(video)
                session.commit()
                session.refresh(video)
            except SQLAlchemyError:
                session.rollback()
                raise

        return video

    def add_scene(
        self,
        video_id: str,
        scene_id: str,
        start_ms: int,
        end_ms: int,
    ) -> Scene:
        """Insert a scene belonging to an existing video."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            scene = Scene(
                video_id=video_id,
                scene_id=scene_id,
                start_ms=start_ms,
                end_ms=end_ms,
            )

            try:
                session.add(scene)
                session.commit()
                session.refresh(scene)
            except SQLAlchemyError:
                session.rollback()
                raise

            return scene

    def add_keyframe(
        self,
        scene_id: str,
        keyframe_id: str,
        visual_index_id: int,
        timestamp_ms: int,
        image_path: str,
    ) -> Keyframe:
        """Insert a keyframe belonging to an existing scene."""

        with self.session_factory() as session:
            if session.get(Scene, scene_id) is None:
                raise ValueError(f"Scene '{scene_id}' does not exist.")

            keyframe = Keyframe(
                scene_id=scene_id,
                keyframe_id=keyframe_id,
                visual_index_id=visual_index_id,
                timestamp_ms=timestamp_ms,
                image_path=image_path,
            )

            try:
                session.add(keyframe)
                session.commit()
                session.refresh(keyframe)
            except SQLAlchemyError:
                session.rollback()
                raise

            return keyframe

    def add_ocr(
        self,
        keyframe_id: str,
        ocr_id: str,
        text: str,
        bbox_x: float,
        bbox_y: float,
        bbox_width: float,
        bbox_height: float,
        confidence: float | None = None,
    ) -> OCR:
        """Insert an OCR record belonging to an existing keyframe."""

        with self.session_factory() as session:
            if session.get(Keyframe, keyframe_id) is None:
                raise ValueError(f"Keyframe '{keyframe_id}' does not exist.")

            ocr = OCR(
                keyframe_id=keyframe_id,
                ocr_id=ocr_id,
                text=text,
                confidence=confidence,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
            )

            try:
                session.add(ocr)
                session.commit()
                session.refresh(ocr)
            except SQLAlchemyError:
                session.rollback()
                raise

            return ocr

    def add_caption(
        self,
        keyframe_id: str,
        caption_id: str,
        text: str,
    ) -> Caption:
        """Insert a caption belonging to an existing keyframe."""

        with self.session_factory() as session:
            if session.get(Keyframe, keyframe_id) is None:
                raise ValueError(f"Keyframe '{keyframe_id}' does not exist.")

            caption = Caption(
                keyframe_id=keyframe_id,
                caption_id=caption_id,
                text=text,
            )

            try:
                session.add(caption)
                session.commit()
                session.refresh(caption)
            except SQLAlchemyError:
                session.rollback()
                raise

            return caption

    def add_transcript_segment(
        self,
        video_id: str,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        text: str,
    ) -> TranscriptSegment:
        """Insert a transcript segment belonging to an existing video."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            transcript_segment = TranscriptSegment(
                video_id=video_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )

            try:
                session.add(transcript_segment)
                session.commit()
                session.refresh(transcript_segment)
            except SQLAlchemyError:
                session.rollback()
                raise

            return transcript_segment

    def add_embedding_record(
        self,
        faiss_id: int,
        index_version: str,
        keyframe_id: str | None = None,
        model_name: str | None = None,
    ) -> EmbeddingRecord:
        """Insert FAISS metadata, optionally linked to a keyframe."""

        with self.session_factory() as session:
            if (
                keyframe_id is not None
                and session.get(Keyframe, keyframe_id) is None
            ):
                raise ValueError(f"Keyframe '{keyframe_id}' does not exist.")

            embedding_record = EmbeddingRecord(
                faiss_id=faiss_id,
                index_version=index_version,
                keyframe_id=keyframe_id,
                model_name=model_name,
            )

            try:
                session.add(embedding_record)
                session.commit()
                session.refresh(embedding_record)
            except SQLAlchemyError:
                session.rollback()
                raise

            return embedding_record
