from __future__ import annotations

import json
from pathlib import Path

from BackEnd.app.pipeline.object_detection_ingestion import (
    ingest_object_detection_jsonl,
    iter_object_detection_rows,
)


class _FakeSession:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: object, rows: list[dict[str, object]]) -> None:
        self.rows.extend(rows)


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def begin(self) -> _FakeSession:
        return self.session


class _FakeDatabase:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.session_factory = _FakeSessionFactory(self.session)


def test_iter_object_detection_rows_uses_official_frame_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "L21_V014" / "141.jsonl"
    source_path.parent.mkdir()
    source_path.write_text(
        json.dumps(
            {
                "video_id": "L21_V014",
                "keyframe_index": 141,
                "confidence": 0.9,
                "class_mid": "/m/01g317",
                "class_name": "Person",
                "class_label": 69,
                "bbox": {
                    "y_min": 0.1,
                    "x_min": 0.2,
                    "y_max": 0.7,
                    "x_max": 0.8,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(
        iter_object_detection_rows(
            root_dir=tmp_path,
            model_name="organizer-openimages",
            model_version="v1",
        )
    )

    assert rows == [
        {
            "frame_id": "L21_V014_141",
            "class_id": "/m/01g317",
            "confidence": 0.9,
            "x_min": 0.2,
            "x_max": 0.8,
            "y_min": 0.1,
            "y_max": 0.7,
            "model_name": "organizer-openimages",
            "model_version": "v1",
        }
    ]


def test_iter_object_detection_rows_filters_video_ids(tmp_path: Path) -> None:
    for video_id in ("L21_V001", "L21_V002"):
        source_path = tmp_path / video_id / "001.jsonl"
        source_path.parent.mkdir()
        source_path.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "keyframe_index": 1,
                    "confidence": 0.5,
                    "class_mid": "/m/01g317",
                    "bbox": {
                        "y_min": 0.1,
                        "x_min": 0.2,
                        "y_max": 0.7,
                        "x_max": 0.8,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    rows = list(
        iter_object_detection_rows(
            root_dir=tmp_path,
            video_ids=["L21_V002"],
            model_name="organizer-openimages",
            model_version="v1",
        )
    )

    assert [row["frame_id"] for row in rows] == ["L21_V002_001"]


def test_ingestion_bulk_inserts_without_frame_lookup(tmp_path: Path) -> None:
    source_path = tmp_path / "L21_V001" / "001.jsonl"
    source_path.parent.mkdir()
    source_path.write_text(
        json.dumps(
            {
                "video_id": "L21_V001",
                "keyframe_index": 1,
                "confidence": 0.5,
                "class_mid": "/m/01g317",
                "bbox": {
                    "y_min": 0.1,
                    "x_min": 0.2,
                    "y_max": 0.7,
                    "x_max": 0.8,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    database = _FakeDatabase()

    inserted = ingest_object_detection_jsonl(
        db=database,  # type: ignore[arg-type]
        root_dir=tmp_path,
        batch_size=1,
        model_name="organizer-openimages",
        model_version="v1",
    )

    assert inserted == 1
    assert database.session.rows[0]["frame_id"] == "L21_V001_001"
