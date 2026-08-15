"""Convert organizer object JSON files to normalized, thresholded JSONL.

Input files contain parallel detection arrays at
``objects/<video_id>/<keyframe_index>.json``. This converter preserves that
tree as ``objects/<video_id>/<keyframe_index>.jsonl`` and writes one detection
per JSONL line for the ObjectDetection ingestion pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from BackEnd.CONFIG import (
    AIC25_OBJECT_JSONL_DIR,
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    ORGANIZER_OBJECT_DIR,
)

_REQUIRED_FIELDS = (
    "detection_scores",
    "detection_class_names",
    "detection_class_entities",
    "detection_boxes",
    "detection_class_labels",
)


def convert_directory(
    *,
    input_dir: Path = ORGANIZER_OBJECT_DIR,
    output_dir: Path = AIC25_OBJECT_JSONL_DIR,
    confidence_threshold: float = OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Convert all organizer JSON detections and return ``(files, records)``.

    Existing output is never replaced unless ``overwrite`` is explicitly set.
    The threshold is inclusive: detections with confidence exactly ``0.25`` are
    retained when the default threshold is used.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1].")

    source_root = Path(input_dir).expanduser().resolve()
    target_root = Path(output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Object JSON directory does not exist: {source_root}")

    source_paths = sorted(source_root.glob("*/*.json"))
    if not source_paths:
        raise FileNotFoundError(f"No object JSON files found in: {source_root}")

    file_count = 0
    record_count = 0
    for source_path in source_paths:
        relative_path = source_path.relative_to(source_root)
        target_path = (target_root / relative_path).with_suffix(".jsonl")
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {target_path}. Use overwrite=True to replace it."
            )

        records = _convert_file(source_path, confidence_threshold)
        _write_jsonl(target_path, records)
        file_count += 1
        record_count += len(records)

    return file_count, record_count


def _convert_file(path: Path, confidence_threshold: float) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Object JSON must be an object: {path}")

    arrays = _parallel_arrays(payload, path)
    video_id = path.parent.name
    try:
        keyframe_index = int(path.stem)
    except ValueError as error:
        raise ValueError(f"Object filename must be numeric: {path}") from error
    if keyframe_index <= 0:
        raise ValueError(f"Object filename must be positive: {path}")

    records: list[dict[str, object]] = []
    for score, class_mid, class_name, box, class_label in zip(
        arrays["detection_scores"],
        arrays["detection_class_names"],
        arrays["detection_class_entities"],
        arrays["detection_boxes"],
        arrays["detection_class_labels"],
        strict=True,
    ):
        confidence = _normalized_float(score, "detection_scores", path)
        if confidence < confidence_threshold:
            continue
        y_min, x_min, y_max, x_max = _box_values(box, path)
        if not isinstance(class_mid, str) or not class_mid:
            raise ValueError(f"Invalid detection_class_names value in {path}")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"Invalid detection_class_entities value in {path}")
        try:
            label = int(class_label)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid detection_class_labels value in {path}") from error

        records.append(
            {
                "video_id": video_id,
                "keyframe_index": keyframe_index,
                "confidence": confidence,
                "class_mid": class_mid,
                "class_name": class_name,
                "class_label": label,
                "bbox": {
                    "y_min": y_min,
                    "x_min": x_min,
                    "y_max": y_max,
                    "x_max": x_max,
                },
            }
        )
    return records


def _parallel_arrays(payload: dict[str, Any], path: Path) -> dict[str, list[Any]]:
    arrays: dict[str, list[Any]] = {}
    for field in _REQUIRED_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list in {path}")
        arrays[field] = value

    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        details = ", ".join(f"{field}={len(values)}" for field, values in arrays.items())
        raise ValueError(f"Detection arrays have inconsistent lengths in {path}: {details}")
    return arrays


def _box_values(box: Any, path: Path) -> tuple[float, float, float, float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"detection_boxes entries must contain four values in {path}")
    y_min, x_min, y_max, x_max = (
        _normalized_float(value, "detection_boxes", path) for value in box
    )
    if y_min >= y_max or x_min >= x_max:
        raise ValueError(f"Invalid detection box ordering in {path}")
    return y_min, x_min, y_max, x_max


def _normalized_float(value: Any, field: str, path: Path) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain numeric values in {path}") from error
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be within [0, 1] in {path}")
    return converted


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert organizer object JSON files to filtered JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=ORGANIZER_OBJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=AIC25_OBJECT_JSONL_DIR)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    files, records = convert_directory(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        confidence_threshold=arguments.confidence_threshold,
        overwrite=arguments.overwrite,
    )
    print(f"Converted {files} files and retained {records} detections.")


if __name__ == "__main__":
    main()
