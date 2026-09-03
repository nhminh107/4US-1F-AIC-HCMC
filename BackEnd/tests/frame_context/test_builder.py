from __future__ import annotations

import json

import pytest

from BackEnd.app.frame_context.artifact import (
    read_frame_context_artifact,
    write_frame_context_artifact,
)
from BackEnd.app.frame_context.builder import build_frame_context
from BackEnd.app.frame_context.contracts import FrameEvidence


def _evidence() -> FrameEvidence:
    return FrameEvidence(
        frame_id="L01_V001_001",
        video_id="L01_V001",
        frame_idx=90,
        timestamp_ms=3_000,
        captions=(" Người đàn ông sửa xe. ", "Người đàn ông sửa xe."),
        ocr_texts=(" HONDA ", "SỬA   XE", "honda"),
        object_labels=("person", "motorcycle", "person"),
    )


def test_build_frame_context_is_deterministic_and_deduplicated() -> None:
    context = build_frame_context(_evidence())

    assert context.caption_text == "Người đàn ông sửa xe."
    assert context.ocr_text == "HONDA; SỬA XE"
    assert context.object_text == "person x2; motorcycle x1"
    assert context.context_text == (
        "[CAPTION]\nNgười đàn ông sửa xe.\n\n"
        "[OCR]\nHONDA; SỬA XE\n\n"
        "[OBJECTS]\nperson x2; motorcycle x1"
    )


def test_empty_evidence_keeps_canonical_identity() -> None:
    evidence = FrameEvidence(
        frame_id="L01_V001_002",
        video_id="L01_V001",
        frame_idx=120,
        timestamp_ms=4_000,
    )

    context = build_frame_context(evidence)

    assert context.context_text == ""
    assert context.frame_idx == 120


def test_artifact_round_trip_and_checksum_validation(tmp_path) -> None:
    record = build_frame_context(_evidence())
    artifact_root = write_frame_context_artifact(
        [record], tmp_path, build_id="context-test"
    )

    assert read_frame_context_artifact(artifact_root) == [record]
    manifest = json.loads((artifact_root / "manifest.json").read_text())
    assert manifest["record_count"] == 1
    assert manifest["searchable_record_count"] == 1

    with pytest.raises(FileExistsError):
        write_frame_context_artifact([record], tmp_path, build_id="context-test")
