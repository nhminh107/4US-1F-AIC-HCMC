from __future__ import annotations

import json

import pytest

from BackEnd.app.pipeline.benchmark_ocr_precision import compare_results


def test_compare_results_reports_speed_and_text_similarity(tmp_path) -> None:
    fp32_path = tmp_path / "fp32.json"
    fp16_path = tmp_path / "fp16.json"
    common = {
        "video_id": "V001",
        "frame_count": 2,
    }
    fp32_path.write_text(
        json.dumps(
            {
                **common,
                "elapsed_seconds": 12.0,
                "row_count": 3,
                "texts_by_frame": {"F1": ["A", "B"], "F2": ["C"]},
            }
        ),
        encoding="utf-8",
    )
    fp16_path.write_text(
        json.dumps(
            {
                **common,
                "elapsed_seconds": 8.0,
                "row_count": 3,
                "texts_by_frame": {"F1": ["A", "B"], "F2": ["D"]},
            }
        ),
        encoding="utf-8",
    )

    report = compare_results(fp32_path, fp16_path)

    assert report["speedup"] == pytest.approx(1.5)
    assert report["exact_frame_ratio"] == pytest.approx(0.5)
    assert report["text_multiset_jaccard"] == pytest.approx(0.5)
