"""Populate PostgreSQL with the complete Open Images V4 class vocabulary."""

from __future__ import annotations

import csv
from pathlib import Path

from BackEnd.CONFIG import OPENIMAGES_CLASS_MAP_PATH
from BackEnd.app.contracts.pipeline import ClassMetadata
from BackEnd.app.database.postgre_db import PostgreManager

OPEN_IMAGES_V4_CLASS_COUNT = 601


def load_open_images_v4_classes(
    class_map_path: Path = OPENIMAGES_CLASS_MAP_PATH,
) -> list[ClassMetadata]:
    """Load and validate the official 601-class Open Images V4 label map."""

    if not class_map_path.is_file():
        raise FileNotFoundError(f"Class map does not exist: {class_map_path}")

    classes_by_label: dict[int, ClassMetadata] = {}
    class_ids: set[str] = set()
    class_names: set[str] = set()
    with class_map_path.open("r", encoding="utf-8", newline="") as class_map_file:
        reader = csv.DictReader(class_map_file)
        expected_fields = {"label", "class_id", "class_name"}
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"Class map must contain columns {sorted(expected_fields)}: "
                f"{class_map_path}"
            )

        for row in reader:
            label = int(row["label"])
            class_id = row["class_id"].strip()
            class_name = row["class_name"].strip()
            if label in classes_by_label:
                raise ValueError(f"Duplicate Open Images label: {label}")
            if class_id in class_ids:
                raise ValueError(f"Duplicate Open Images class_id: {class_id}")
            if class_name in class_names:
                raise ValueError(f"Duplicate Open Images class_name: {class_name}")
            if len(class_id) > 15:
                raise ValueError(f"ClassID exceeds varchar(15): '{class_id}'")
            if len(class_name) > 50:
                raise ValueError(f"Class name exceeds varchar(50): '{class_name}'")

            classes_by_label[label] = ClassMetadata(class_id, class_name)
            class_ids.add(class_id)
            class_names.add(class_name)

    expected_labels = set(range(1, OPEN_IMAGES_V4_CLASS_COUNT + 1))
    actual_labels = set(classes_by_label)
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        unexpected = sorted(actual_labels - expected_labels)
        raise ValueError(
            "Open Images V4 class map must contain labels 1 through 601; "
            f"missing={missing}, unexpected={unexpected}."
        )

    return [classes_by_label[label] for label in sorted(classes_by_label)]


def ingest_class_ids(
    db: PostgreManager,
    class_map_path: Path = OPENIMAGES_CLASS_MAP_PATH,
) -> list[ClassMetadata]:
    """Insert all 601 Open Images V4 classes into PostgreSQL."""

    classes = load_open_images_v4_classes(class_map_path)
    for metadata in classes:
        db.add_class_id(metadata.class_id, metadata.class_name)
    return classes


def main() -> None:
    classes = ingest_class_ids(PostgreManager())
    print(f"ClassID ingestion completed: {len(classes)} classes")


if __name__ == "__main__":
    main()
