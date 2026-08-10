"""Tests for organizer keyframe discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from BackEnd.app.pipeline.official_keyframe_ingestion import load_official_keyframes


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    map_dir = tmp_path / "maps"
    object_dir = tmp_path / "objects"
    keyframe_dir = tmp_path / "keyframes"
    map_dir.mkdir()
    (object_dir / "L21_V001").mkdir(parents=True)
    (keyframe_dir / "L21_V001").mkdir(parents=True)
    (map_dir / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0.0,30.0,0\n"
        "2,0.01,30.0,0\n",
        encoding="utf-8",
    )
    (object_dir / "L21_V001" / "001.json").write_text("{}", encoding="utf-8")
    (object_dir / "L21_V001" / "002.json").write_text("{}", encoding="utf-8")
    (keyframe_dir / "L21_V001" / "001.jpg").write_bytes(b"local-image")
    return map_dir, object_dir, keyframe_dir


def test_load_preserves_all_logical_frames_and_fractional_pts(tmp_path: Path) -> None:
    map_dir, object_dir, keyframe_dir = _dataset(tmp_path)

    frames = load_official_keyframes(map_dir, object_dir, keyframe_dir)

    assert [frame.frame_id for frame in frames] == ["L21_V001_001", "L21_V001_002"]
    assert [frame.frame_idx for frame in frames] == [0, 0]
    assert frames[1].pts_time == 0.01
    assert frames[1].timestamp_ms == 10
    assert frames[0].frame_path == keyframe_dir / "L21_V001" / "001.jpg"
    assert frames[1].frame_path == keyframe_dir / "L21_V001" / "002.jpg"
    assert all(frame.source == "official" for frame in frames)
    assert all(frame.shot_id is None for frame in frames)


def test_load_rejects_map_and_object_coverage_mismatch(tmp_path: Path) -> None:
    map_dir, object_dir, keyframe_dir = _dataset(tmp_path)
    (object_dir / "L21_V001" / "002.json").unlink()

    with pytest.raises(ValueError, match="Keyframe coverage differs"):
        load_official_keyframes(map_dir, object_dir, keyframe_dir)
