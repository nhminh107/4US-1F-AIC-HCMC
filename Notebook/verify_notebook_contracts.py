"""Static contract audit for every committed Kaggle notebook.

The check deliberately does not execute notebook cells, download models, or
open a database.  It verifies the minimum offline hand-off: configuration for
an input dataset and an SQL INSERT artifact for the target table.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SQL = {
    "Caption/caption_fpt_vlm.ipynb": "INSERT INTO Caption",
    "ShotExtractor/shot_extractor_kaggle.ipynb": "INSERT INTO shot",
    "KeyframeExtractor/keyframe_extractor_kaggle.ipynb": "INSERT INTO frame",
    "ClipExtractor/clip_extractor_kaggle.ipynb": "INSERT INTO clipwindow",
    "ObjectDetection/object_detection_ingestion_kaggle.ipynb": "INSERT INTO objectdetection",
    "Tracking/tracking_yolo26_bytetrack_kaggle.ipynb": "INSERT INTO objecttrack",
    "OCR/ocr_kaggle.ipynb": "INSERT INTO ocr",
    "FrameEmbedding/frame_embedding_siglip_kaggle.ipynb": "INSERT INTO frameembeddingrecord",
    "ClipEmbedding/clip_embedding_siglip_kaggle.ipynb": "INSERT INTO clipembeddingrecord",
    "ShotEmbedding/shot_embedding_siglip_kaggle.ipynb": "INSERT INTO shotembeddingrecord",
}


def source_of(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def main() -> None:
    failures: list[str] = []
    for relative, sql_marker in EXPECTED_SQL.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"Missing notebook: {relative}")
            continue
        source = source_of(path)
        if sql_marker.casefold() not in source.casefold():
            failures.append(f"{relative}: missing SQL marker {sql_marker!r}")
        if "input" not in source.casefold():
            failures.append(f"{relative}: no visible input configuration")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Notebook contracts passed: {len(EXPECTED_SQL)}")


if __name__ == "__main__":
    main()
