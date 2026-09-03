"""Load frame specialist evidence from PostgreSQL."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from sqlalchemy import Engine, inspect, text

from BackEnd.app.frame_context.contracts import FrameEvidence


def load_frame_evidence(
    engine: Engine,
    *,
    video_ids: Iterable[str] | None = None,
    minimum_object_confidence: float = 0.25,
) -> list[FrameEvidence]:
    """Load frame, caption, OCR, and object evidence without mutating the database."""

    if not 0.0 <= minimum_object_confidence <= 1.0:
        raise ValueError("minimum_object_confidence must be within [0, 1].")

    selected_video_ids = list(video_ids or _load_video_ids(engine))
    evidence: list[FrameEvidence] = []
    for video_id in selected_video_ids:
        evidence.extend(
            _load_video_evidence(
                engine,
                video_id,
                minimum_object_confidence=minimum_object_confidence,
            )
        )
    return evidence


def _load_video_ids(engine: Engine) -> list[str]:
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT video_id FROM video ORDER BY video_id"))
        return [str(row.video_id) for row in rows]


def _load_video_evidence(
    engine: Engine,
    video_id: str,
    *,
    minimum_object_confidence: float,
) -> list[FrameEvidence]:
    with engine.connect() as connection:
        frames = connection.execute(
            text(
                "SELECT frame_id, video_id, shot_id, frame_idx, timestamp_ms "
                "FROM frame WHERE video_id = :video_id "
                "ORDER BY timestamp_ms, frame_id"
            ),
            {"video_id": video_id},
        ).mappings().all()
        frame_ids = {str(frame["frame_id"]) for frame in frames}

        ocr_by_frame: dict[str, list[str]] = defaultdict(list)
        for row in connection.execute(
            text(
                "SELECT o.frame_id, o.text FROM ocr o "
                "JOIN frame f ON f.frame_id = o.frame_id "
                "WHERE f.video_id = :video_id ORDER BY o.frame_id, o.n"
            ),
            {"video_id": video_id},
        ):
            ocr_by_frame[str(row.frame_id)].append(str(row.text))

        objects_by_frame: dict[str, list[str]] = defaultdict(list)
        for row in connection.execute(
            text(
                "SELECT d.frame_id, c.class_name FROM objectdetection d "
                "JOIN frame f ON f.frame_id = d.frame_id "
                "JOIN classid c ON c.class_id = d.class_id "
                "WHERE f.video_id = :video_id AND d.confidence >= :confidence "
                "ORDER BY d.frame_id, c.class_name"
            ),
            {"video_id": video_id, "confidence": minimum_object_confidence},
        ):
            objects_by_frame[str(row.frame_id)].append(str(row.class_name))

        captions_by_frame = _load_captions(connection, frames, frame_ids, video_id)

    return [
        FrameEvidence(
            frame_id=str(frame["frame_id"]),
            video_id=str(frame["video_id"]),
            frame_idx=int(frame["frame_idx"]),
            timestamp_ms=int(frame["timestamp_ms"]),
            captions=tuple(captions_by_frame.get(str(frame["frame_id"]), ())),
            ocr_texts=tuple(ocr_by_frame.get(str(frame["frame_id"]), ())),
            object_labels=tuple(objects_by_frame.get(str(frame["frame_id"]), ())),
        )
        for frame in frames
    ]


def _load_captions(connection, frames, frame_ids: set[str], video_id: str):
    """Support both the generic ORM schema and the current shot-only live schema."""

    caption_columns = {
        column["name"] for column in inspect(connection).get_columns("caption")
    }
    captions_by_frame: dict[str, list[str]] = defaultdict(list)
    if "frame_id" in caption_columns:
        rows = connection.execute(
            text(
                "SELECT c.frame_id, c.caption_text FROM caption c "
                "JOIN frame f ON f.frame_id = c.frame_id "
                "WHERE c.frame_id IS NOT NULL AND f.video_id = :video_id "
                "ORDER BY c.caption_id"
            ),
            {"video_id": video_id},
        )
        for row in rows:
            frame_id = str(row.frame_id)
            if frame_id in frame_ids:
                captions_by_frame[frame_id].append(str(row.caption_text))

    if "shot_id" not in caption_columns:
        return captions_by_frame

    shot_captions: dict[str, list[str]] = defaultdict(list)
    rows = connection.execute(
        text(
            "SELECT c.shot_id, c.caption_text FROM caption c "
            "JOIN shot s ON s.shot_id = c.shot_id "
            "WHERE c.shot_id IS NOT NULL AND s.video_id = :video_id "
            "ORDER BY c.caption_id"
        ),
        {"video_id": video_id},
    )
    for row in rows:
        shot_captions[str(row.shot_id)].append(str(row.caption_text))

    shots = connection.execute(
        text(
            "SELECT shot_id, start_ms, end_ms FROM shot "
            "WHERE video_id = :video_id ORDER BY start_ms"
        ),
        {"video_id": video_id},
    ).mappings().all()
    for frame in frames:
        frame_id = str(frame["frame_id"])
        shot_id = frame["shot_id"]
        if shot_id is None:
            timestamp_ms = int(frame["timestamp_ms"])
            matching_shot = next(
                (
                    shot
                    for shot in shots
                    if int(shot["start_ms"]) <= timestamp_ms < int(shot["end_ms"])
                ),
                None,
            )
            shot_id = matching_shot["shot_id"] if matching_shot else None
        if shot_id is not None:
            captions_by_frame[frame_id].extend(shot_captions.get(str(shot_id), ()))
    return captions_by_frame
