"""Bulk-ingest organizer Open Images JSONL detections into PostgreSQL."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
from typing import Any

from sqlalchemy import insert

from BackEnd.CONFIG import (
    AIC25_OBJECT_JSONL_DIR,
    OPENIMAGES_MODEL_NAME,
    OPENIMAGES_MODEL_VERSION,
    POSTGRES_INSERT_BATCH_SIZE,
)
from BackEnd.app.database.models import ObjectDetection
from BackEnd.app.database.postgre_db import PostgreManager


def ingest_object_detection_jsonl(
    *,
    db: PostgreManager,
    root_dir: Path = AIC25_OBJECT_JSONL_DIR,
    batch_size: int = POSTGRES_INSERT_BATCH_SIZE,
    video_ids: Iterable[str] | None = None,
    model_name: str = OPENIMAGES_MODEL_NAME,
    model_version: str = OPENIMAGES_MODEL_VERSION,
) -> int:
    """Insert organizer object detections without querying the ``Frame`` table.

    ``Frame`` and ``ClassID`` records must already exist. Their foreign keys are
    enforced by PostgreSQL during insertion; this function intentionally does
    not perform per-record existence checks.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rows = iter_object_detection_rows(
        root_dir=root_dir,
        video_ids=video_ids,
        model_name=model_name,
        model_version=model_version,
    )
    inserted = 0
    batch: list[dict[str, object]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            _insert_batch(db, batch)
            inserted += len(batch)
            batch.clear()

    if batch:
        _insert_batch(db, batch)
        inserted += len(batch)

    return inserted


def iter_object_detection_rows(
    *,
    root_dir: Path,
    video_ids: Iterable[str] | None = None,
    model_name: str,
    model_version: str,
) -> Iterator[dict[str, object]]:
    """Yield database rows from organizer JSONL files in deterministic order."""

    resolved_root = Path(root_dir).expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Object JSONL directory does not exist: {resolved_root}")

    selected_video_ids = set(video_ids) if video_ids is not None else None
    for path in sorted(resolved_root.glob("*/*.jsonl")):
        video_id = path.parent.name
        if selected_video_ids is not None and video_id not in selected_video_ids:
            continue

        keyframe_index = _keyframe_index(path)
        frame_id = f"{video_id}_{keyframe_index:03d}"
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = _load_json_record(path, line_number, line)
                yield _to_object_detection_row(
                    record,
                    path=path,
                    line_number=line_number,
                    frame_id=frame_id,
                    video_id=video_id,
                    keyframe_index=keyframe_index,
                    model_name=model_name,
                    model_version=model_version,
                )


def _insert_batch(db: PostgreManager, rows: list[dict[str, object]]) -> None:
    """Insert one bounded batch in its own transaction."""

    with db.session_factory.begin() as session:
        session.execute(insert(ObjectDetection), rows)


def _keyframe_index(path: Path) -> int:
    try:
        keyframe_index = int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Invalid keyframe filename: {path}") from exc
    if keyframe_index <= 0:
        raise ValueError(f"Keyframe index must be positive: {path}")
    return keyframe_index


def _load_json_record(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"JSON record must be an object at {path}:{line_number}")
    return record


def _to_object_detection_row(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    frame_id: str,
    video_id: str,
    keyframe_index: int,
    model_name: str,
    model_version: str,
) -> dict[str, object]:
    if record.get("video_id") != video_id:
        raise ValueError(f"video_id mismatch at {path}:{line_number}")
    if record.get("keyframe_index") != keyframe_index:
        raise ValueError(f"keyframe_index mismatch at {path}:{line_number}")

    bbox = record.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"bbox must be an object at {path}:{line_number}")

    class_id = _string_value(record, "class_mid", path, line_number)
    confidence = _normalized_value(record, "confidence", path, line_number)
    x_min = _normalized_value(bbox, "x_min", path, line_number)
    x_max = _normalized_value(bbox, "x_max", path, line_number)
    y_min = _normalized_value(bbox, "y_min", path, line_number)
    y_max = _normalized_value(bbox, "y_max", path, line_number)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid bbox ordering at {path}:{line_number}")

    return {
        "frame_id": frame_id,
        "class_id": class_id,
        "confidence": confidence,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "model_name": model_name,
        "model_version": model_version,
    }


def _string_value(
    record: dict[str, Any],
    field: str,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string at {path}:{line_number}")
    return value


def _normalized_value(
    record: dict[str, Any],
    field: str,
    path: Path,
    line_number: int,
) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric at {path}:{line_number}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be within [0, 1] at {path}:{line_number}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest organizer object detections into PostgreSQL."
    )
    parser.add_argument("--root-dir", type=Path, default=AIC25_OBJECT_JSONL_DIR)
    parser.add_argument("--batch-size", type=int, default=POSTGRES_INSERT_BATCH_SIZE)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--model-name", default=OPENIMAGES_MODEL_NAME)
    parser.add_argument("--model-version", default=OPENIMAGES_MODEL_VERSION)
    arguments = parser.parse_args()

    inserted = ingest_object_detection_jsonl(
        db=PostgreManager(),
        root_dir=arguments.root_dir,
        batch_size=arguments.batch_size,
        video_ids=arguments.video_ids,
        model_name=arguments.model_name,
        model_version=arguments.model_version,
    )
    print(f"Inserted {inserted} ObjectDetection records.")


if __name__ == "__main__":
    main()
