from __future__ import annotations

import json
from pathlib import Path

import pytest

from BackEnd.app.pipeline.embedding_only import _validate_ocr_result


def test_validate_ocr_result_accepts_exact_success_count(tmp_path: Path) -> None:
    result_path = tmp_path / "ocr-result.json"
    result_path.write_text(
        json.dumps(
            {
                "completed_video_ids": ["L01_V001", "L01_V002"],
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    _validate_ocr_result(result_path, expected_completed_videos=2)


def test_validate_ocr_result_rejects_partial_success(tmp_path: Path) -> None:
    result_path = tmp_path / "ocr-result.json"
    result_path.write_text(
        json.dumps({"completed_video_ids": ["L01_V001"], "error": None}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected=2, actual=1"):
        _validate_ocr_result(result_path, expected_completed_videos=2)
