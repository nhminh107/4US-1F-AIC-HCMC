"""Ingest organizer-provided video metadata into PostgreSQL."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from BackEnd.CONFIG import (
    PROJECT_ROOT,
    VIDEO_DESCRIPTION_MAX_LENGTH as DESCRIPTION_MAX_LENGTH,
    VIDEO_DIR as DEFAULT_VIDEO_DIR,
    VIDEO_METADATA_DIR as DEFAULT_METADATA_DIR,
)
from BackEnd.app.database.models import Video
from BackEnd.app.database.postgre_db import PostgreManager

def ingest_videos(
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
    video_dir: str | Path = DEFAULT_VIDEO_DIR,
    database_url: str | None = None,
) -> dict[str, int]:
    """Insert missing video records from organizer metadata into PostgreSQL.

    Existing ``video_id`` values are skipped, making the function safe to rerun.
    A video record is still created when its local MP4 is unavailable because
    organizer metadata covers more videos than the local media collection.

    Returns counts for discovered, inserted, skipped, locally available, and
    truncated-description records.
    """

    resolved_metadata_dir = Path(metadata_dir).expanduser().resolve()
    resolved_video_dir = Path(video_dir).expanduser().resolve()
    if not resolved_metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Metadata directory does not exist: {resolved_metadata_dir}"
        )

    metadata_paths = sorted(resolved_metadata_dir.glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No JSON metadata files found in: {resolved_metadata_dir}"
        )

    manager = PostgreManager(database_url=database_url)
    inserted = 0
    skipped = 0
    local_videos = 0
    truncated_descriptions = 0
    session = None

    try:
        with manager.session_factory() as session:
            existing_video_ids = set(session.scalars(select(Video.video_id)).all())

            for metadata_path in metadata_paths:
                video_id = metadata_path.stem
                video_file = resolved_video_dir / f"{video_id}.mp4"
                if video_file.is_file():
                    local_videos += 1

                if video_id in existing_video_ids:
                    skipped += 1
                    continue

                try:
                    with metadata_path.open("r", encoding="utf-8") as file:
                        metadata = json.load(file)
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Cannot read video metadata: {metadata_path}"
                    ) from error

                try:
                    stored_video_path = video_file.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    stored_video_path = str(video_file)

                description = metadata.get("description")
                if description and len(description) > DESCRIPTION_MAX_LENGTH:
                    description = description[:DESCRIPTION_MAX_LENGTH]
                    truncated_descriptions += 1

                publish_date = metadata.get("publish_date")
                parsed_publish_date = (
                    datetime.strptime(publish_date, "%d/%m/%Y").date()
                    if publish_date
                    else None
                )

                duration_seconds = metadata.get("length")
                duration_ms = (
                    int(duration_seconds) * 1000
                    if duration_seconds is not None
                    else None
                )

                session.add(
                    Video(
                        video_id=video_id,
                        video_path=stored_video_path,
                        title=metadata.get("title"),
                        description=description,
                        keywords=metadata.get("keywords"),
                        author=metadata.get("author"),
                        channel_id=metadata.get("channel_id"),
                        channel_url=metadata.get("channel_url"),
                        watch_url=metadata.get("watch_url"),
                        thumbnail_url=metadata.get("thumbnail_url"),
                        publish_date=parsed_publish_date,
                        duration_ms=duration_ms,
                    )
                )
                existing_video_ids.add(video_id)
                inserted += 1

            session.commit()
    except (OSError, TypeError, ValueError, SQLAlchemyError):
        if session is not None:
            session.rollback()
        raise
    finally:
        manager.engine.dispose()

    if truncated_descriptions:
        warnings.warn(
            f"Truncated {truncated_descriptions} video descriptions to "
            f"{DESCRIPTION_MAX_LENGTH} characters to match the database schema.",
            stacklevel=2,
        )

    return {
        "discovered": len(metadata_paths),
        "inserted": inserted,
        "skipped": skipped,
        "local_videos": local_videos,
        "truncated_descriptions": truncated_descriptions,
    }


if __name__ == "__main__":
    print(ingest_videos())
