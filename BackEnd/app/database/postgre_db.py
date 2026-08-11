"""PostgreSQL connection management and common persistence operations."""

from __future__ import annotations

from dataclasses import replace
import os
from datetime import date
from typing import Any, TypeVar

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from BackEnd.app.database.adapter import (
    clip_embedding_mapping_from_record,
    clip_window_metadata_from_clip_window,
    frame_embedding_mapping_from_record,
    frame_metadata_from_frame,
    shot_embedding_mapping_from_record,
    shot_metadata_from_shot,
    video_metadata_from_video,
)
from BackEnd.app.database.models import (
    Base,
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
    TranscriptSegment,
    TrackObservation,
    Video,
)

from BackEnd.app.contracts.pipeline import (
    ClipEmbeddingMapping,
    ClipWindowMetadata,
    FrameEmbeddingMapping,
    FrameMetadata,
    ObjectTrackResult,
    ShotEmbeddingMapping,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
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
        shot_id: str | None,
        timestamp_ms: int,
        fps: float,
        frame_idx: int,
        source: str,
        *,
        n: int | None = None,
        pts_time: float | None = None,
        frame_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Frame:
        """Insert a frame, optionally belonging to an existing shot."""

        with self.session_factory() as session:
            if source == "official" and shot_id is not None:
                raise ValueError("Official frames must have shot_id=None.")
            if shot_id is not None:
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
                source=source,
                frame_path=frame_path,
                width=width,
                height=height,
            )
            return self._persist(session, frame)

    def add_clip(
        self,
        clip_id: str,
        shot_id: str,
        start_ms: int,
        end_ms: int,
        *,
        start_frame_idx: int | None = None,
        end_frame_idx: int | None = None,
        sampling_fps: float | None = None,
        clip_path: str | None = None,
    ) -> ClipWindow:
        """Insert a clip window belonging to an existing shot."""

        with self.session_factory() as session:
            if session.get(Shot, shot_id) is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")

            clip = ClipWindow(
                clip_id=clip_id,
                shot_id=shot_id,
                start_ms=start_ms,
                end_ms=end_ms,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                sampling_fps=sampling_fps,
                clip_path=clip_path,
            )
            return self._persist(session, clip)

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

    def add_class_id(self, class_id: str, class_name: str) -> ClassID:
        """Insert one object class, or return the matching existing record."""

        with self.session_factory() as session:
            existing = session.get(ClassID, class_id)
            if existing is not None:
                if existing.class_name != class_name:
                    raise ValueError(
                        f"ClassID '{class_id}' already has name "
                        f"'{existing.class_name}', not '{class_name}'."
                    )
                return existing

            class_record = ClassID(class_id=class_id, class_name=class_name)
            return self._persist(session, class_record)

    def add_object_detection(
        self,
        frame_id: str,
        class_id: str,
        confidence: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> ObjectDetection:
        """Insert one object detection belonging to an existing frame."""

        with self.session_factory() as session:
            if session.get(Frame, frame_id) is None:
                raise ValueError(f"Frame '{frame_id}' does not exist.")
            if session.get(ClassID, class_id) is None:
                raise ValueError(f"ClassID '{class_id}' does not exist.")

            detection = ObjectDetection(
                frame_id=frame_id,
                class_id=class_id,
                confidence=confidence,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                model_name=model_name,
                model_version=model_version,
            )
            return self._persist(session, detection)

    def add_object_track(
        self,
        shot_id: str,
        class_id: str,
        start_ms: int,
        end_ms: int,
        observation_count: int,
        *,
        model_name: str,
        model_version: str,
        tracker_name: str,
        tracker_version: str,
        sampling_fps: float,
        mapping_version: str,
        start_frame_idx: int | None = None,
        end_frame_idx: int | None = None,
        avg_confidence: float | None = None,
    ) -> ObjectTrack:
        """Insert one object track belonging to an existing shot."""

        with self.session_factory() as session:
            if session.get(Shot, shot_id) is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")
            if session.get(ClassID, class_id) is None:
                raise ValueError(f"ClassID '{class_id}' does not exist.")

            track = ObjectTrack(
                shot_id=shot_id,
                class_id=class_id,
                start_ms=start_ms,
                end_ms=end_ms,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                observation_count=observation_count,
                avg_confidence=avg_confidence,
                model_name=model_name,
                model_version=model_version,
                tracker_name=tracker_name,
                tracker_version=tracker_version,
                sampling_fps=sampling_fps,
                mapping_version=mapping_version,
            )
            return self._persist(session, track)

    def add_track_observation(
        self,
        track_id: int,
        frame_idx: int,
        timestamp_ms: int,
        confidence: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> TrackObservation:
        """Insert one normalized observation belonging to an object track."""

        with self.session_factory() as session:
            if session.get(ObjectTrack, track_id) is None:
                raise ValueError(f"ObjectTrack '{track_id}' does not exist.")

            observation = TrackObservation(
                track_id=track_id,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                confidence=confidence,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            return self._persist(session, observation)

    def add_tracking_result(
        self,
        tracks: list[ObjectTrackResult],
        observations: list[TrackObservationResult],
    ) -> list[ObjectTrackResult]:
        """Insert tracks and their observations in one transaction."""

        if not tracks:
            if observations:
                raise ValueError("Tracking observations require at least one track.")
            return []

        with self.session_factory.begin() as session:
            track_records = [
                ObjectTrack(
                    shot_id=track.shot_id,
                    class_id=track.class_id,
                    start_ms=track.start_ms,
                    end_ms=track.end_ms,
                    start_frame_idx=track.start_frame_idx,
                    end_frame_idx=track.end_frame_idx,
                    observation_count=track.observation_count,
                    avg_confidence=track.avg_confidence,
                    model_name=track.model_name,
                    model_version=track.model_version,
                    tracker_name=track.tracker_name,
                    tracker_version=track.tracker_version,
                    sampling_fps=track.sampling_fps,
                    mapping_version=track.mapping_version,
                )
                for track in tracks
            ]
            session.add_all(track_records)
            session.flush()

            track_id_map: dict[int, int] = {}
            for track, record in zip(tracks, track_records):
                if track.track_id is None:
                    raise ValueError("Tracking result must have a local track_id.")
                track_id_map[track.track_id] = record.track_id

            try:
                observation_records = [
                    TrackObservation(
                        track_id=track_id_map[observation.track_id],
                        frame_idx=observation.frame_idx,
                        timestamp_ms=observation.timestamp_ms,
                        confidence=observation.confidence,
                        x_min=observation.x_min,
                        x_max=observation.x_max,
                        y_min=observation.y_min,
                        y_max=observation.y_max,
                    )
                    for observation in observations
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Observation references unknown local track_id: {exc.args[0]}."
                ) from exc
            session.add_all(observation_records)

        return [
            replace(track, track_id=record.track_id)
            for track, record in zip(tracks, track_records)
        ]

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

    def add_clip_embedding_record(
        self,
        faiss_id: int,
        index_version: int,
        clip_id: str,
        model_name: str,
    ) -> ClipEmbeddingRecord:
        """Insert a FAISS mapping for an existing clip window."""

        with self.session_factory() as session:
            if session.get(ClipWindow, clip_id) is None:
                raise ValueError(f"Clip '{clip_id}' does not exist.")

            record = ClipEmbeddingRecord(
                faiss_id=faiss_id,
                index_version=index_version,
                clip_id=clip_id,
                model_name=model_name,
            )
            return self._persist(session, record)

    def add_shot_embedding_record(
        self,
        faiss_id: int,
        index_version: int,
        shot_id: str,
        model_name: str,
        model_version: str,
        *,
        pooling_method: str = "mean",
    ) -> ShotEmbeddingRecord:
        """Insert a FAISS mapping for an existing shot."""

        with self.session_factory() as session:
            if session.get(Shot, shot_id) is None:
                raise ValueError(f"Shot '{shot_id}' does not exist.")

            record = ShotEmbeddingRecord(
                faiss_id=faiss_id,
                index_version=index_version,
                shot_id=shot_id,
                model_name=model_name,
                model_version=model_version,
                pooling_method=pooling_method,
            )
            return self._persist(session, record)

    def get_frame_embedding_mappings(
        self,
        frame_ids: list[str],
        *,
        index_version: int,
        model_name: str,
    ) -> list[FrameEmbeddingMapping]:
        """Return existing frame mappings for one FAISS/model version."""

        if not frame_ids:
            return []
        with self.session_factory() as session:
            records = session.scalars(
                select(FrameEmbeddingRecord).where(
                    FrameEmbeddingRecord.frame_id.in_(frame_ids),
                    FrameEmbeddingRecord.index_version == index_version,
                    FrameEmbeddingRecord.model_name == model_name,
                )
            ).all()
            return [frame_embedding_mapping_from_record(record) for record in records]

    def get_clip_embedding_mappings(
        self,
        clip_ids: list[str],
        *,
        index_version: int,
        model_name: str,
    ) -> list[ClipEmbeddingMapping]:
        """Return existing clip mappings for one FAISS/model version."""

        if not clip_ids:
            return []
        with self.session_factory() as session:
            records = session.scalars(
                select(ClipEmbeddingRecord).where(
                    ClipEmbeddingRecord.clip_id.in_(clip_ids),
                    ClipEmbeddingRecord.index_version == index_version,
                    ClipEmbeddingRecord.model_name == model_name,
                )
            ).all()
            return [clip_embedding_mapping_from_record(record) for record in records]

    def get_shot_embedding_mappings(
        self,
        shot_ids: list[str],
        *,
        index_version: int,
        model_name: str,
        model_version: str,
        pooling_method: str,
    ) -> list[ShotEmbeddingMapping]:
        """Return existing shot mappings for one FAISS/model version."""

        if not shot_ids:
            return []
        with self.session_factory() as session:
            records = session.scalars(
                select(ShotEmbeddingRecord).where(
                    ShotEmbeddingRecord.shot_id.in_(shot_ids),
                    ShotEmbeddingRecord.index_version == index_version,
                    ShotEmbeddingRecord.model_name == model_name,
                    ShotEmbeddingRecord.model_version == model_version,
                    ShotEmbeddingRecord.pooling_method == pooling_method,
                )
            ).all()
            return [shot_embedding_mapping_from_record(record) for record in records]

    def add_frame_embedding_records(
        self,
        mappings: list[FrameEmbeddingMapping],
    ) -> None:
        """Insert frame mappings in one PostgreSQL transaction."""

        if not mappings:
            return
        frame_ids = {mapping.frame_id for mapping in mappings}
        with self.session_factory.begin() as session:
            existing_ids = set(
                session.scalars(select(Frame.frame_id).where(Frame.frame_id.in_(frame_ids)))
            )
            missing_ids = sorted(frame_ids - existing_ids)
            if missing_ids:
                raise ValueError(f"Frames do not exist: {missing_ids[:10]}.")
            session.add_all(
                [
                    FrameEmbeddingRecord(
                        faiss_id=mapping.faiss_id,
                        index_version=mapping.index_version,
                        frame_id=mapping.frame_id,
                        model_name=mapping.model_name,
                    )
                    for mapping in mappings
                ]
            )

    def add_clip_embedding_records(
        self,
        mappings: list[ClipEmbeddingMapping],
    ) -> None:
        """Insert clip mappings in one PostgreSQL transaction."""

        if not mappings:
            return
        clip_ids = {mapping.clip_id for mapping in mappings}
        with self.session_factory.begin() as session:
            existing_ids = set(
                session.scalars(
                    select(ClipWindow.clip_id).where(ClipWindow.clip_id.in_(clip_ids))
                )
            )
            missing_ids = sorted(clip_ids - existing_ids)
            if missing_ids:
                raise ValueError(f"Clips do not exist: {missing_ids[:10]}.")
            session.add_all(
                [
                    ClipEmbeddingRecord(
                        faiss_id=mapping.faiss_id,
                        index_version=mapping.index_version,
                        clip_id=mapping.clip_id,
                        model_name=mapping.model_name,
                    )
                    for mapping in mappings
                ]
            )

    def add_shot_embedding_records(
        self,
        mappings: list[ShotEmbeddingMapping],
    ) -> None:
        """Insert shot mappings in one PostgreSQL transaction."""

        if not mappings:
            return
        shot_ids = {mapping.shot_id for mapping in mappings}
        with self.session_factory.begin() as session:
            existing_ids = set(
                session.scalars(select(Shot.shot_id).where(Shot.shot_id.in_(shot_ids)))
            )
            missing_ids = sorted(shot_ids - existing_ids)
            if missing_ids:
                raise ValueError(f"Shots do not exist: {missing_ids[:10]}.")
            session.add_all(
                [
                    ShotEmbeddingRecord(
                        faiss_id=mapping.faiss_id,
                        index_version=mapping.index_version,
                        shot_id=mapping.shot_id,
                        model_name=mapping.model_name,
                        model_version=mapping.model_version,
                        pooling_method=mapping.pooling_method,
                    )
                    for mapping in mappings
                ]
            )

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

    def get_list_video(self) -> list[VideoMetadata]:
        with self.session_factory() as session:
            statement = (
                select(Video).order_by(Video.video_id)
            )

            Videos = session.scalars(statement).all()

            return [video_metadata_from_video(vid) for vid in Videos]
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
    print(db.get_list_video())


if __name__ == "__main__":
    main()
