from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from BackEnd.app.pipeline.embedding_ingestion import iter_organizer_embedding_batches


def _write_keyframe_map(path: Path, numbers: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["n"])
        writer.writeheader()
        for number in numbers:
            writer.writerow({"n": number})


def test_embedding_rows_map_to_official_frame_ids(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    map_dir = tmp_path / "maps"
    feature_dir.mkdir()
    map_dir.mkdir()
    np.save(
        feature_dir / "L21_V001.npy",
        np.asarray([[1.0] * 512, [2.0] * 512], dtype=np.float16),
    )
    _write_keyframe_map(map_dir / "L21_V001.csv", [1, 2])

    batches = list(
        iter_organizer_embedding_batches(
            feature_dir=feature_dir,
            map_dir=map_dir,
            batch_size=1,
        )
    )

    assert [references[0].frame_id for _, references in batches] == [
        "L21_V001_001",
        "L21_V001_002",
    ]
    assert batches[0][0].dtype == np.float32
    assert batches[1][0][0, 0] == 2.0


def test_embedding_ingestion_rejects_non_sequential_keyframe_numbers(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    map_dir = tmp_path / "maps"
    feature_dir.mkdir()
    map_dir.mkdir()
    np.save(feature_dir / "L21_V001.npy", np.zeros((2, 512), dtype=np.float16))
    _write_keyframe_map(map_dir / "L21_V001.csv", [1, 3])

    with pytest.raises(ValueError, match="Expected n=2"):
        list(
            iter_organizer_embedding_batches(
                feature_dir=feature_dir,
                map_dir=map_dir,
                batch_size=8,
            )
        )
