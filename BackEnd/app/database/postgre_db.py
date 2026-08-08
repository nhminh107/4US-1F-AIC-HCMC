"""PostgreSQL connection management and common persistence operations."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, TypeVar

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from BackEnd.app.database.adapter import (
    clip_window_metadata_from_clip_window,
    frame_metadata_from_frame,
    shot_metadata_from_shot,
)
from BackEnd.app.database.models import (
    Base,
    Caption,
    ClipWindow,
    Frame,
    FrameEmbeddingRecord,
    OCR,
    Shot,
    TranscriptSegment,
    Video,
)

from BackEnd.app.contracts.pipeline import (
    ClipWindowMetadata,
    FrameMetadata,
    ShotMetadata,
)

load_dotenv()

_ModelT = TypeVar("_ModelT", bound=Base)


class PostgreManager:
    """Manage PostgreSQL sessions and records used by the pipeline."""

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

    @staticmethod
    def _persist(session: Session, record: _ModelT) -> _ModelT:
        """Commit one ORM record and return it with database values loaded."""

        try:
            session.add(record)
            session.commit()
            session.refresh(record)
        except SQLAlchemyError:
            session.rollback()
            raise
        return record

    def add_video(
        self,
        video_id: str,
        video_path: str,
        *,
        title: str | None = None,
        description: str | None = None,
        keywords: list[str] | None = None,
        author: str | None = None,
        channel_id: str | None = None,
        channel_url: str | None = None,
        watch_url: str | None = None,
        thumbnail_url: str | None = None,
        publish_date: date | None = None,
        duration_ms: int | None = None,
    ) -> Video:
        """Insert a source video and return the persisted record."""

        video = Video(
            video_id=video_id,
            video_path=video_path,
            title=title,
            description=description,
            keywords=keywords,
            author=author,
            channel_id=channel_id,
            channel_url=channel_url,
            watch_url=watch_url,
            thumbnail_url=thumbnail_url,
            publish_date=publish_date,
            duration_ms=duration_ms,
        )
        with self.session_factory() as session:
            return self._persist(session, video)

    def add_shot(
        self,
        shot_id: str,
        video_id: str,
        shot_index: int,
        start_ms: int,
        end_ms: int,
        *,
        start_frame_idx: int | None = None,
        end_frame_idx: int | None = None,
    ) -> Shot:
        """Insert a shot belonging to an existing video."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            shot = Shot(
                shot_id=shot_id,
                video_id=video_id,
                shot_index=shot_index,
                start_ms=start_ms,
                end_ms=end_ms,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
            )
            return self._persist(session, shot)

    def add_frame(
        self,
        frame_id: str,
        video_id: str,
        shot_id: str,
        timestamp_ms: int,
        fps: float,
        frame_idx: int,
        frame_role: str,
        source: str,
        *,
        n: int | None = None,
        pts_time: int | None = None,
        frame_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Frame:
        """Insert a frame belonging to an existing shot."""

        with self.session_factory() as session:
            shot = session.get(Shot, shot_id)
            if shot is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")
            if shot.video_id != video_id:
                raise ValueError(
                    f"Shot '{shot_id}' does not belong to video '{video_id}'."
                )

            frame = Frame(
                frame_id=frame_id,
                n=n,
                video_id=video_id,
                shot_id=shot_id,
                pts_time=pts_time,
                timestamp_ms=timestamp_ms,
                fps=fps,
                frame_idx=frame_idx,
                frame_role=frame_role,
                source=source,
                frame_path=frame_path,
                width=width,
                height=height,
            )
            return self._persist(session, frame)

    def add_ocr(
        self,
        frame_id: str,
        n: int,
        text: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        language: str | None = None,
    ) -> OCR:
        """Insert an OCR result belonging to an existing frame."""

        with self.session_factory() as session:
            if session.get(Frame, frame_id) is None:
                raise ValueError(f"Frame '{frame_id}' does not exist.")

            ocr = OCR(
                frame_id=frame_id,
                n=n,
                text=text,
                language=language,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            return self._persist(session, ocr)

    def add_caption(
        self,
        caption_text: str,
        model_name: str,
        *,
        frame_id: str | None = None,
        clip_id: str | None = None,
        shot_id: str | None = None,
        structured_data: dict[str, Any] | None = None,
        model_version: str | None = None,
        prompt_version: str | None = None,
        confidence: float | None = None,
    ) -> Caption:
        """Insert a caption targeting exactly one frame, clip, or shot."""

        target_count = sum(
            target_id is not None for target_id in (frame_id, clip_id, shot_id)
        )
        if target_count != 1:
            raise ValueError(
                "A caption must reference exactly one frame, clip, or shot."
            )

        caption = Caption(
            frame_id=frame_id,
            clip_id=clip_id,
            shot_id=shot_id,
            caption_text=caption_text,
            structured_data=structured_data,
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            confidence=confidence,
        )
        with self.session_factory() as session:
            return self._persist(session, caption)

    def add_transcript_segment(
        self,
        segment_id: str,
        video_id: str,
        start_ms: int,
        end_ms: int,
        text: str,
        *,
        language: str | None = None,
    ) -> TranscriptSegment:
        """Insert a transcript segment belonging to an existing video."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            segment = TranscriptSegment(
                segment_id=segment_id,
                video_id=video_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                language=language,
            )
            return self._persist(session, segment)

    def add_frame_embedding_record(
        self,
        faiss_id: int,
        index_version: int,
        frame_id: str,
        model_name: str,
    ) -> FrameEmbeddingRecord:
        """Insert a FAISS mapping for an existing frame."""

        with self.session_factory() as session:
            if session.get(Frame, frame_id) is None:
                raise ValueError(f"Frame '{frame_id}' does not exist.")

            record = FrameEmbeddingRecord(
                faiss_id=faiss_id,
                index_version=index_version,
                frame_id=frame_id,
                model_name=model_name,
            )
            return self._persist(session, record)

    def get_frame_record_by_frame_id(self, frame_id: str) -> FrameMetadata:
        with self.session_factory() as session:
            data = session.get(Frame, frame_id)
            if data is None:
                raise ValueError(f"Frame '{frame_id}' does not exist.")

            return frame_metadata_from_frame(data)

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        """Return all frames belonging to a video, ordered by frame index."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            statement = (
                select(Frame)
                .where(Frame.video_id == video_id)
                .order_by(Frame.frame_idx)
            )
            frames = session.scalars(statement).all()

            return [frame_metadata_from_frame(frame) for frame in frames]

    def get_list_frame_in_shot(self, shot_id: str) -> list[FrameMetadata]:
        """Return all frames belonging to a shot, ordered by frame index."""

        with self.session_factory() as session:
            if session.get(Shot, shot_id) is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")

            statement = (
                select(Frame)
                .where(Frame.shot_id == shot_id)
                .order_by(Frame.frame_idx)
            )
            frames = session.scalars(statement).all()

            return [frame_metadata_from_frame(frame) for frame in frames]

    def get_list_shot_in_video(self, video_id: str) -> list[ShotMetadata]:
        """Return all shots belonging to a video, ordered by shot index."""

        with self.session_factory() as session:
            if session.get(Video, video_id) is None:
                raise ValueError(f"Video '{video_id}' does not exist.")

            statement = (
                select(Shot)
                .where(Shot.video_id == video_id)
                .order_by(Shot.shot_index)
            )
            shots = session.scalars(statement).all()

            return [shot_metadata_from_shot(shot) for shot in shots]

    def get_list_clip_in_shot(self, shot_id: str) -> list[ClipWindowMetadata]:
        """Return all clips belonging to a shot, ordered by start time."""

        with self.session_factory() as session:
            if session.get(Shot, shot_id) is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")

            statement = (
                select(ClipWindow)
                .where(ClipWindow.shot_id == shot_id)
                .order_by(ClipWindow.start_ms)
            )
            clips = session.scalars(statement).all()

            return [
                clip_window_metadata_from_clip_window(clip) for clip in clips
            ]

# Backward-compatible name for existing imports.
#Postgre_Manager = PostgreManager


def main() -> None:
    """Fetch and print one frame to verify the get operation."""

    db = PostgreManager()
    # db.add_video("1", "1")
    # db.add_shot("1", "1", 0, 0, 1000)
    # db.add_frame(
    #     "1",
    #     "1",
    #     "1",
    #     0,
    #     30.0,
    #     0,
    #     "keyframe",
    #     "extracted",
    # )
    print(db.get_frame_record_by_frame_id("1"))


if __name__ == "__main__":
    main()
