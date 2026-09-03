from __future__ import annotations

import json

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytest

from BackEnd.app.text_embedding.builder import build_text_index, validate_text_index
from BackEnd.app.text_embedding.contracts import TextDocument


class FakeEncoder:
    model_id = "fake-text-encoder"
    model_revision = "test-revision"

    def encode(self, texts, *, batch_size: int):
        assert batch_size > 0
        return np.asarray(
            [[len(text), index + 1, 1.0] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


def _documents() -> list[TextDocument]:
    return [
        TextDocument(
            source_type="frame_context",
            entity_id="F1",
            frame_id="F1",
            video_id="V1",
            text="người sửa xe",
            frame_idx=10,
            timestamp_ms=1_000,
        ),
        TextDocument(
            source_type="frame_context",
            entity_id="F2",
            frame_id="F2",
            video_id="V1",
            text="cửa hàng Honda",
            frame_idx=20,
            timestamp_ms=2_000,
        ),
    ]


def test_builds_valid_normalized_flat_ip_index(tmp_path) -> None:
    root = build_text_index(
        _documents(), FakeEncoder(), tmp_path, build_id="context-index-test"
    )

    validation = validate_text_index(root)
    index = faiss.read_index(str(root / "index.faiss"))
    mapping = pq.read_table(root / "mapping.parquet").to_pylist()
    manifest = json.loads((root / "manifest.json").read_text())

    assert validation["valid"] is True
    assert index.ntotal == 2
    assert index.d == 3
    assert [row["entity_id"] for row in mapping] == ["F1", "F2"]
    assert manifest["model_revision"] == "test-revision"


def test_rejects_mixed_sources(tmp_path) -> None:
    documents = _documents()
    documents.append(
        TextDocument(
            source_type="asr_segment",
            entity_id="S1",
            segment_id="S1",
            video_id="V1",
            text="xin chào",
            start_ms=0,
            end_ms=1000,
        )
    )

    with pytest.raises(ValueError, match="one source_type"):
        build_text_index(documents, FakeEncoder(), tmp_path, build_id="mixed")
