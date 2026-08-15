"""Tests for complete Open Images V4 ClassID ingestion."""

from __future__ import annotations

from unittest.mock import patch

from BackEnd.app.contracts.pipeline import ClassMetadata
from BackEnd.app.pipeline.class_id_ingestion import (
    ingest_class_ids,
    load_open_images_v4_classes,
)


class _FakeDatabase:
    def __init__(self) -> None:
        self.classes: list[tuple[str, str]] = []

    def add_class_id(self, class_id: str, class_name: str) -> None:
        self.classes.append((class_id, class_name))


def test_official_class_map_contains_all_601_classes() -> None:
    classes = load_open_images_v4_classes()
    classes_by_id = {item.class_id: item.class_name for item in classes}

    assert len(classes) == 601
    assert classes_by_id["/m/01g317"] == "Person"
    assert classes_by_id["/m/04yx4"] == "Man"
    assert classes_by_id["/m/09j2d"] == "Clothing"
    assert classes_by_id["/m/0xzly"] == "Maracas"


def test_ingest_class_ids_calls_database_for_every_contract() -> None:
    classes = [
        ClassMetadata("/m/01g317", "Person"),
        ClassMetadata("/m/09j2d", "Clothing"),
    ]
    db = _FakeDatabase()

    with patch(
        "BackEnd.app.pipeline.class_id_ingestion.load_open_images_v4_classes",
        return_value=classes,
    ):
        results = ingest_class_ids(db)

    assert results == classes
    assert db.classes == [
        ("/m/01g317", "Person"),
        ("/m/09j2d", "Clothing"),
    ]
