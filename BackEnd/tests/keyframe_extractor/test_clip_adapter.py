"""Tests for the image-only CLIP adapter bridge."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from BackEnd.app.keyframe_extractor.clip_adapter import ClipImageEmbeddingAdapter


@dataclass(frozen=True)
class FakeMetadata:
    model_name: str
    model_revision: str | None


class FakeClipAdapter:
    def __init__(self, vectors) -> None:
        self.vectors = vectors
        self.calls = []

    def encode_images(self, images, batch_size=64):
        self.calls.append((tuple(images), batch_size))
        return self.vectors

    def get_dimension(self):
        return len(self.vectors[0])

    def metadata(self):
        return FakeMetadata(model_name="fake-clip", model_revision="rev1")


class ClipAdapterTests(unittest.TestCase):
    def test_delegates_to_shared_clip_adapter(self) -> None:
        fake = FakeClipAdapter([[1.0, 0.0], [0.0, 1.0]])
        adapter = ClipImageEmbeddingAdapter(fake, batch_size=8)

        vectors = adapter.encode_images(["a", "b"])

        self.assertEqual(vectors.shape, (2, 2))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0)
        self.assertEqual(fake.calls[0][1], 8)
        self.assertEqual(adapter.model_name, "fake-clip")
        self.assertEqual(adapter.model_version, "rev1")
        self.assertEqual(adapter.embedding_dim, 2)

    def test_rejects_wrong_row_count(self) -> None:
        fake = FakeClipAdapter(np.zeros((1, 2), dtype=np.float32))
        adapter = ClipImageEmbeddingAdapter(fake)

        with self.assertRaises(ValueError):
            adapter.encode_images(["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
