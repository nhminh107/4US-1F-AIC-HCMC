"""Convert FrameContext and PostgreSQL ASR data into text documents."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Engine, text

from BackEnd.app.frame_context.artifact import read_frame_context_artifact
from BackEnd.app.text_embedding.contracts import TextDocument


def context_documents(artifact_root) -> list[TextDocument]:
    """Load non-empty FrameContext records for dense embedding."""

    return [
        TextDocument(
            source_type="frame_context",
            entity_id=record.frame_id,
            video_id=record.video_id,
            text=record.context_text,
            frame_id=record.frame_id,
            frame_idx=record.frame_idx,
            timestamp_ms=record.timestamp_ms,
        )
        for record in read_frame_context_artifact(artifact_root)
        if record.context_text.strip()
    ]


def asr_documents(
    engine: Engine,
    *,
    video_ids: Iterable[str] | None = None,
) -> list[TextDocument]:
    """Load non-empty, timestamped ASR segments from PostgreSQL."""

    filters = []
    parameters: dict[str, object] = {}
    selected_video_ids = list(video_ids or ())
    if selected_video_ids:
        placeholders = []
        for index, video_id in enumerate(selected_video_ids):
            name = f"video_id_{index}"
            placeholders.append(f":{name}")
            parameters[name] = video_id
        filters.append(f"video_id IN ({', '.join(placeholders)})")
    filters.append("trim(text) <> ''")
    where_clause = " AND ".join(filters)
    query = text(
        "SELECT segment_id, video_id, start_ms, end_ms, text "
        f"FROM transcriptsegment WHERE {where_clause} "
        "ORDER BY video_id, start_ms, segment_id"
    )
    with engine.connect() as connection:
        rows = connection.execute(query, parameters).mappings().all()
    return [
        TextDocument(
            source_type="asr_segment",
            entity_id=str(row["segment_id"]),
            video_id=str(row["video_id"]),
            text=str(row["text"]).strip(),
            segment_id=str(row["segment_id"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
        )
        for row in rows
    ]
