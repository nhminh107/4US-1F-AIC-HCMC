"""Online retrieval adapters for Elasticsearch and PostgreSQL evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from BackEnd.app.contracts.online_retrieval import (
    ObjectConstraint,
    ObjectRetrievalRequest,
    RetrievalCandidate,
)
from BackEnd.app.contracts.search import TextSearchQuery
from BackEnd.app.database.models import (
    ClassID,
    Frame,
    ObjectDetection,
    ObjectTrack,
    Shot,
)

if TYPE_CHECKING:
    from BackEnd.app.service.text_search_service import TextSearchService


class _TextSearchService(Protocol):
    """Narrow interface used by the online text retrieval adapter."""

    def search(self, query: TextSearchQuery) -> list:
        """Return text hits for one query."""


def _bounded_score(value: float | None) -> float:
    """Convert backend-specific scores into the common [0, 1] interval."""

    if value is None:
        return 0.0
    value = float(value)
    if value <= 0.0:
        return 0.0
    # Elasticsearch scores are unbounded. This monotonic transform preserves rank.
    return value / (1.0 + value)


class TextRetrievalTool:
    """Normalize Elasticsearch hits for candidate fusion."""

    def __init__(self, service: _TextSearchService | None = None) -> None:
        if service is None:
            # Keep Elasticsearch and dotenv optional for tests and callers that
            # inject another backend implementing the same search interface.
            from BackEnd.app.service.text_search_service import TextSearchService

            service = TextSearchService()
        self.service = service

    def search(self, query: TextSearchQuery) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        for hit in self.service.search(query):
            entity_type = "frame" if hit.frame_id else "clip" if hit.clip_id else "shot" if hit.shot_id else "video"
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"text:{hit.doc_id}",
                    source=hit.source_type,
                    entity_type=entity_type,
                    entity_id=hit.entity_id,
                    video_id=hit.video_id,
                    score=_bounded_score(hit.score),
                    shot_id=hit.shot_id,
                    frame_id=hit.frame_id,
                    clip_id=hit.clip_id,
                    timestamp_ms=hit.timestamp_ms,
                    start_ms=hit.start_ms,
                    end_ms=hit.end_ms,
                    evidence=hit.content,
                )
            )
        return candidates


class ObjectTrackingRetrievalTool:
    """Query persisted object detections and tracks without rerunning models."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def search(self, request: ObjectRetrievalRequest) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        with self.session_factory() as session:
            for constraint in request.objects:
                if request.include_detections:
                    candidates.extend(self._search_detections(session, request, constraint))
                if request.include_tracks:
                    candidates.extend(self._search_tracks(session, request, constraint))

        # Stable IDs prevent duplicate evidence when equivalent constraints are sent.
        unique = {candidate.candidate_id: candidate for candidate in candidates}
        return sorted(
            unique.values(),
            key=lambda candidate: (-candidate.score, candidate.video_id, candidate.candidate_id),
        )[: request.top_k]

    @staticmethod
    def _search_detections(
        session: Session,
        request: ObjectRetrievalRequest,
        constraint: ObjectConstraint,
    ) -> list[RetrievalCandidate]:
        statement = (
            select(ObjectDetection, Frame, ClassID)
            .join(Frame, ObjectDetection.frame_id == Frame.frame_id)
            .join(ClassID, ObjectDetection.class_id == ClassID.class_id)
            .where(ClassID.class_name.ilike(constraint.class_name))
            .where(ObjectDetection.confidence >= constraint.minimum_confidence)
        )
        if request.video_ids:
            statement = statement.where(Frame.video_id.in_(request.video_ids))
        if request.start_ms is not None:
            statement = statement.where(Frame.timestamp_ms >= request.start_ms)
        if request.end_ms is not None:
            statement = statement.where(Frame.timestamp_ms < request.end_ms)
        statement = statement.order_by(ObjectDetection.confidence.desc()).limit(request.top_k)

        return [
            RetrievalCandidate(
                candidate_id=f"detection:{detection.detection_id}",
                source="object_detection",
                entity_type="frame",
                entity_id=frame.frame_id,
                video_id=frame.video_id,
                score=float(detection.confidence),
                shot_id=frame.shot_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                evidence=f"Detected {object_class.class_name}",
                class_id=object_class.class_id,
                class_name=object_class.class_name,
                metadata=(("model_name", detection.model_name or "unknown"),),
            )
            for detection, frame, object_class in session.execute(statement).all()
        ]

    @staticmethod
    def _search_tracks(
        session: Session,
        request: ObjectRetrievalRequest,
        constraint: ObjectConstraint,
    ) -> list[RetrievalCandidate]:
        statement = (
            select(ObjectTrack, Shot, ClassID)
            .join(Shot, ObjectTrack.shot_id == Shot.shot_id)
            .join(ClassID, ObjectTrack.class_id == ClassID.class_id)
            .where(ClassID.class_name.ilike(constraint.class_name))
            .where(ObjectTrack.avg_confidence >= constraint.minimum_confidence)
            .where(
                ObjectTrack.end_ms - ObjectTrack.start_ms
                >= constraint.minimum_track_duration_ms
            )
        )
        if request.video_ids:
            statement = statement.where(Shot.video_id.in_(request.video_ids))
        if request.start_ms is not None:
            statement = statement.where(ObjectTrack.end_ms > request.start_ms)
        if request.end_ms is not None:
            statement = statement.where(ObjectTrack.start_ms < request.end_ms)
        statement = statement.order_by(ObjectTrack.avg_confidence.desc()).limit(request.top_k)

        return [
            RetrievalCandidate(
                candidate_id=f"track:{track.track_id}",
                source="object_track",
                entity_type="track",
                entity_id=str(track.track_id),
                video_id=shot.video_id,
                score=float(track.avg_confidence or 0.0),
                shot_id=track.shot_id,
                start_ms=track.start_ms,
                end_ms=track.end_ms,
                evidence=(
                    f"Tracked {object_class.class_name} across "
                    f"{track.observation_count} observations"
                ),
                class_id=object_class.class_id,
                class_name=object_class.class_name,
                metadata=(
                    ("model_name", track.model_name),
                    ("tracker_name", track.tracker_name),
                ),
            )
            for track, shot, object_class in session.execute(statement).all()
        ]


__all__ = ["ObjectTrackingRetrievalTool", "TextRetrievalTool"]
