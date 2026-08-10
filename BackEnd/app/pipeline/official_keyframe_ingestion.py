"""Insert organizer-provided keyframe metadata into PostgreSQL.

The organizer object tree defines the complete logical keyframe coverage while
``map-keyframes`` provides the canonical time and frame-index metadata. Every
frame keeps its deterministic expected JPG path, even when the file is not yet
available locally.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

from sqlalchemy import inspect, select
from sqlalchemy.dialects.postgresql import insert

from BackEnd.CONFIG import (
    KEYFRAME_MAP_DIR,
    KEYFRAME_OUTPUT_DIR,
    ORGANIZER_OBJECT_DIR,
    POSTGRES_INSERT_BATCH_SIZE,
    PROJECT_ROOT,
)
from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.database.models import Frame, Video
from BackEnd.app.database.postgre_db import PostgreManager


_VIDEO_ID_PATTERN = re.compile(r"L\d+_V\d+")


def load_official_keyframes(
    map_dir: str | Path = KEYFRAME_MAP_DIR,
    object_dir: str | Path = ORGANIZER_OBJECT_DIR,
    keyframe_dir: str | Path = KEYFRAME_OUTPUT_DIR,
    *,
    video_ids: Iterable[str] | None = None,
) -> list[FrameMetadata]:
    """Return all validated organizer keyframes as pipeline contracts.

    Organizer-provided frames deliberately have no shot assignment. Their
    ``shot_id`` remains ``None`` throughout ingestion.
    """

    resolved_map_dir = Path(map_dir).expanduser().resolve()
    resolved_object_dir = Path(object_dir).expanduser().resolve()
    resolved_keyframe_dir = Path(keyframe_dir).expanduser().resolve()
    _require_directory(resolved_map_dir, "keyframe map")
    _require_directory(resolved_object_dir, "organizer object")

    map_paths = {path.stem: path for path in resolved_map_dir.glob("*.csv")}
    object_dirs = {
        path.name: path for path in resolved_object_dir.iterdir() if path.is_dir()
    }
    if set(map_paths) != set(object_dirs):
        missing_maps = sorted(set(object_dirs) - set(map_paths))
        missing_objects = sorted(set(map_paths) - set(object_dirs))
        raise ValueError(
            "Map/object video coverage differs: "
            f"missing_maps={missing_maps[:10]} "
            f"missing_object_dirs={missing_objects[:10]}."
        )

    selected_video_ids = set(video_ids) if video_ids is not None else set(map_paths)
    unknown_video_ids = selected_video_ids - set(map_paths)
    if unknown_video_ids:
        raise ValueError(f"Unknown video IDs: {sorted(unknown_video_ids)}")

    frames: list[FrameMetadata] = []
    for video_id in sorted(selected_video_ids):
        if _VIDEO_ID_PATTERN.fullmatch(video_id) is None:
            raise ValueError(f"Invalid organizer video ID: {video_id}.")

        rows = _read_map_rows(map_paths[video_id])
        map_numbers = {int(row["n"]) for row in rows}
        object_numbers = _read_object_numbers(object_dirs[video_id])
        if map_numbers != object_numbers:
            raise ValueError(
                f"Keyframe coverage differs for {video_id}: "
                f"map_only={sorted(map_numbers - object_numbers)[:10]} "
                f"object_only={sorted(object_numbers - map_numbers)[:10]}."
            )

        for row in rows:
            n = _positive_int(row["n"], "n", video_id)
            pts_time = _non_negative_float(row["pts_time"], "pts_time", video_id)
            fps = _positive_float(row["fps"], "fps", video_id)
            frame_idx = _non_negative_int(row["frame_idx"], "frame_idx", video_id)
            frame_id = f"{video_id}_{n:03d}"
            if len(frame_id) > 15:
                raise ValueError(f"Frame ID exceeds varchar(15): {frame_id}.")

            expected_image = resolved_keyframe_dir / video_id / f"{n:03d}.jpg"
            frame_path = _stored_local_path(expected_image)
            frames.append(
                FrameMetadata(
                    frame_id=frame_id,
                    video_id=video_id,
                    shot_id=None,
                    timestamp_ms=round(pts_time * 1_000),
                    fps=fps,
                    frame_idx=frame_idx,
                    source="official",
                    n=n,
                    pts_time=pts_time,
                    frame_path=frame_path,
                )
            )

    return frames


def ingest_official_keyframes(
    frames: Sequence[FrameMetadata],
    *,
    database_url: str | None = None,
    batch_size: int = POSTGRES_INSERT_BATCH_SIZE,
) -> dict[str, int]:
    """Upsert official keyframe contracts and return insertion statistics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if not frames:
        return {
            "discovered": 0,
            "local_images": 0,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
        }

    _validate_contracts(frames)
    manager = PostgreManager(database_url=database_url)
    try:
        _validate_database_frame_schema(manager)
        with manager.session_factory.begin() as session:
            video_ids = sorted({frame.video_id for frame in frames})
            videos = {
                video.video_id: video
                for video in session.scalars(
                    select(Video).where(Video.video_id.in_(video_ids))
                )
            }
            missing_video_ids = sorted(set(video_ids) - set(videos))
            if missing_video_ids:
                raise ValueError(
                    "Videos must be ingested before their keyframes. Missing video IDs: "
                    f"{missing_video_ids[:20]}."
                )

            existing_frames = session.scalars(
                select(Frame).where(Frame.video_id.in_(video_ids))
            ).all()
            existing_by_id = {frame.frame_id: frame for frame in existing_frames}
            existing_official_by_number = {
                (frame.video_id, frame.n): frame
                for frame in existing_frames
                if frame.source == "official"
            }
            _reject_identity_conflicts(
                frames,
                existing_by_id,
                existing_official_by_number,
            )

            pending: list[FrameMetadata] = []
            inserted = 0
            updated = 0
            unchanged = 0
            for frame in frames:
                existing = existing_by_id.get(frame.frame_id)
                if existing is None:
                    inserted += 1
                    pending.append(frame)
                elif _frame_values(existing) == _contract_values(frame):
                    unchanged += 1
                else:
                    updated += 1
                    pending.append(frame)

            for start in range(0, len(pending), batch_size):
                payload = [_frame_payload(frame) for frame in pending[start : start + batch_size]]
                statement = insert(Frame).values(payload)
                statement = statement.on_conflict_do_update(
                    index_elements=[Frame.frame_id],
                    set_={
                        "n": statement.excluded.n,
                        "video_id": statement.excluded.video_id,
                        "shot_id": statement.excluded.shot_id,
                        "pts_time": statement.excluded.pts_time,
                        "timestamp_ms": statement.excluded.timestamp_ms,
                        "fps": statement.excluded.fps,
                        "frame_idx": statement.excluded.frame_idx,
                        "source": statement.excluded.source,
                        "frame_path": statement.excluded.frame_path,
                        "width": statement.excluded.width,
                        "height": statement.excluded.height,
                    },
                )
                session.execute(statement)

        return {
            "discovered": len(frames),
            "local_images": sum(_frame_path_exists(frame) for frame in frames),
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
        }
    finally:
        manager.engine.dispose()


def _read_map_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_fields = {"n", "pts_time", "fps", "frame_idx"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_fields:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {sorted(required_fields)}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Keyframe map is empty: {path}.")
    return rows


def _read_object_numbers(path: Path) -> set[int]:
    numbers: set[int] = set()
    for object_path in path.glob("*.json"):
        if not object_path.stem.isdigit():
            raise ValueError(f"Object filename must be numeric: {object_path}.")
        number = int(object_path.stem)
        if number in numbers:
            raise ValueError(f"Duplicate object keyframe number in {path}: {number}.")
        numbers.add(number)
    if not numbers:
        raise ValueError(f"No organizer object JSON files found in {path}.")
    return numbers


def _validate_contracts(frames: Sequence[FrameMetadata]) -> None:
    frame_ids: set[str] = set()
    official_numbers: set[tuple[str, int]] = set()
    for frame in frames:
        if frame.source != "official":
            raise ValueError(f"Frame is not organizer-provided: {frame.frame_id}.")
        if frame.shot_id is not None:
            raise ValueError(f"Official frame must have shot_id=None: {frame.frame_id}.")
        if frame.n is None or frame.n < 1:
            raise ValueError(f"Official frame n must start at 1: {frame.frame_id}.")
        if frame.pts_time is None or frame.pts_time < 0:
            raise ValueError(f"Invalid pts_time for {frame.frame_id}.")
        if frame.timestamp_ms != round(frame.pts_time * 1_000):
            raise ValueError(f"timestamp_ms does not match pts_time: {frame.frame_id}.")
        if frame.fps <= 0 or frame.frame_idx < 0:
            raise ValueError(f"Invalid frame timing metadata: {frame.frame_id}.")
        if frame.frame_id in frame_ids:
            raise ValueError(f"Duplicate frame_id: {frame.frame_id}.")
        official_number = (frame.video_id, frame.n)
        if official_number in official_numbers:
            raise ValueError(f"Duplicate organizer (video_id, n): {official_number}.")
        frame_ids.add(frame.frame_id)
        official_numbers.add(official_number)


def _validate_database_frame_schema(manager: PostgreManager) -> None:
    inspector = inspect(manager.engine)
    columns = {column["name"]: column for column in inspector.get_columns("frame")}
    if "frame_role" in columns:
        raise RuntimeError(
            "The database still contains Frame.frame_role. Apply the schema migration "
            "that drops this column before running organizer keyframe ingestion."
        )
    if not columns.get("shot_id", {}).get("nullable", False):
        raise RuntimeError(
            "Frame.shot_id is still NOT NULL. Apply the official keyframe schema "
            "migration before ingestion."
        )
    pts_type = str(columns.get("pts_time", {}).get("type", "")).lower()
    if "int" in pts_type:
        raise RuntimeError(
            "Frame.pts_time is still an integer column. Migrate it to FLOAT before "
            "ingestion so organizer fractional PTS values are not lost."
        )
    unique_columns = {
        tuple(constraint.get("column_names") or ())
        for constraint in inspector.get_unique_constraints("frame")
    }
    if ("video_id", "frame_idx") in unique_columns:
        raise RuntimeError(
            "The database still enforces unique(video_id, frame_idx), but the "
            "organizer maps contain repeated frame_idx values. Apply the official "
            "keyframe schema migration before ingestion."
        )
    indexes = {index["name"]: index for index in inspector.get_indexes("frame")}
    if "uq_frame_official_video_n" not in indexes:
        raise RuntimeError(
            "Missing partial unique index uq_frame_official_video_n. Apply the "
            "official keyframe schema migration before ingestion."
        )


def _reject_identity_conflicts(
    frames: Sequence[FrameMetadata],
    existing_by_id: dict[str, Frame],
    existing_official_by_number: dict[tuple[str, int | None], Frame],
) -> None:
    for frame in frames:
        same_id = existing_by_id.get(frame.frame_id)
        if same_id is not None and (
            same_id.video_id != frame.video_id or same_id.frame_idx != frame.frame_idx
        ):
            raise ValueError(
                f"Existing frame_id {frame.frame_id} points to a different frame."
            )
        same_number = existing_official_by_number.get((frame.video_id, frame.n))
        if same_number is not None and same_number.frame_id != frame.frame_id:
            raise ValueError(
                f"Existing organizer ({frame.video_id}, {frame.n}) uses frame_id "
                f"{same_number.frame_id}, expected {frame.frame_id}."
            )


def _frame_payload(frame: FrameMetadata) -> dict[str, object]:
    return {
        "frame_id": frame.frame_id,
        "n": frame.n,
        "video_id": frame.video_id,
        "shot_id": frame.shot_id,
        "pts_time": frame.pts_time,
        "timestamp_ms": frame.timestamp_ms,
        "fps": frame.fps,
        "frame_idx": frame.frame_idx,
        "source": frame.source,
        "frame_path": str(frame.frame_path) if frame.frame_path is not None else None,
        "width": frame.width,
        "height": frame.height,
    }


def _frame_values(frame: Frame) -> tuple[object, ...]:
    return (
        frame.n,
        frame.video_id,
        frame.shot_id,
        frame.pts_time,
        frame.timestamp_ms,
        frame.fps,
        frame.frame_idx,
        frame.source,
        frame.frame_path,
        frame.width,
        frame.height,
    )


def _contract_values(frame: FrameMetadata) -> tuple[object, ...]:
    payload = _frame_payload(frame)
    return tuple(
        payload[name]
        for name in (
            "n",
            "video_id",
            "shot_id",
            "pts_time",
            "timestamp_ms",
            "fps",
            "frame_idx",
            "source",
            "frame_path",
            "width",
            "height",
        )
    )


def _stored_local_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _frame_path_exists(frame: FrameMetadata) -> bool:
    if frame.frame_path is None:
        return False
    path = frame.frame_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.is_file()


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label.capitalize()} directory does not exist: {path}.")


def _positive_int(value: str, field: str, video_id: str) -> int:
    parsed = _non_negative_int(value, field, video_id)
    if parsed == 0:
        raise ValueError(f"{field} must be positive for {video_id}.")
    return parsed


def _non_negative_int(value: str, field: str, video_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}={value!r} for {video_id}.") from error
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative for {video_id}.")
    return parsed


def _positive_float(value: str, field: str, video_id: str) -> float:
    parsed = _non_negative_float(value, field, video_id)
    if parsed == 0:
        raise ValueError(f"{field} must be positive for {video_id}.")
    return parsed


def _non_negative_float(value: str, field: str, video_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}={value!r} for {video_id}.") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative for {video_id}.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert all organizer-provided keyframe metadata into PostgreSQL."
    )
    parser.add_argument("--map-dir", type=Path, default=KEYFRAME_MAP_DIR)
    parser.add_argument("--object-dir", type=Path, default=ORGANIZER_OBJECT_DIR)
    parser.add_argument("--keyframe-dir", type=Path, default=KEYFRAME_OUTPUT_DIR)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--batch-size", type=int, default=POSTGRES_INSERT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count local data without connecting to PostgreSQL.",
    )
    arguments = parser.parse_args()

    frames = load_official_keyframes(
        map_dir=arguments.map_dir,
        object_dir=arguments.object_dir,
        keyframe_dir=arguments.keyframe_dir,
        video_ids=arguments.video_ids,
    )
    if arguments.dry_run:
        print(
            {
                "discovered": len(frames),
                "videos": len({frame.video_id for frame in frames}),
                "local_images": sum(_frame_path_exists(frame) for frame in frames),
            }
        )
        return

    print(ingest_official_keyframes(frames, batch_size=arguments.batch_size))


if __name__ == "__main__":
    main()
