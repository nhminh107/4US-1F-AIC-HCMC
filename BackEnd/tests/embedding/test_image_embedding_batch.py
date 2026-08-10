"""Tests for frame batch embedding preprocessing and dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.embedding.BaseEmbedding import BaseEmbedder
from BackEnd.app.embedding.ImageEmbedding import ImageEmbedder


class _BatchEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.encoded_batch: object | None = None

    def get_real_data(self, data):
        return data

    def preprocess(self, data):
        return data

    def encode(self, data):
        raise AssertionError("embed_batch must not call encode")

    def get_real_data_list(self, batch_data):
        return list(batch_data)

    def preprocess_batch(self, batch_data):
        return batch_data

    def encode_batch(self, batch_data):
        self.encoded_batch = batch_data
        return np.asarray([[1.0]], dtype=np.float32)


def test_base_embed_batch_dispatches_to_encode_batch() -> None:
    embedder = _BatchEmbedder()

    result = embedder.embed_batch(["frame"])

    assert result.shape == (1, 1)
    assert embedder.encoded_batch == ["frame"]


def test_image_embedder_resolves_relative_paths_from_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "data" / "frame.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 3), color="white").save(image_path)
    monkeypatch.setattr(
        "BackEnd.app.embedding.ImageEmbedding.cf.PROJECT_ROOT",
        tmp_path,
    )
    frame = FrameMetadata(
        frame_id="L21_V005_E001",
        video_id="L21_V005",
        shot_id="L21_V005_S000",
        timestamp_ms=0,
        fps=30.0,
        frame_idx=0,
        source="extracted",
        frame_path=Path("data/frame.jpg"),
    )
    embedder = object.__new__(ImageEmbedder)

    image = embedder.get_real_data(frame)

    assert image.size == (4, 3)
