from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.embedding.models.clip_vit_b32 import ClipViTB32Adapter


class FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, values, *, batch_size, convert_to_numpy, show_progress_bar):
        self.batch_sizes.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        return np.ones((len(values), 4), dtype=np.float32)


class ClipAdapterTests(unittest.TestCase):
    def test_encode_texts_normalizes_and_uses_oom_fallback(self) -> None:
        model = FakeModel()
        adapter = ClipViTB32Adapter(model=model)

        vectors = adapter.encode_texts(["hello", "world"], batch_size=8)

        self.assertEqual(model.batch_sizes, [8, 4, 2])
        self.assertEqual(vectors.shape, (2, 4))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1], atol=1e-6)

    def test_encode_texts_rejects_empty_text(self) -> None:
        adapter = ClipViTB32Adapter(model=FakeModel())

        with self.assertRaisesRegex(ValueError, "non-empty"):
            adapter.encode_texts([" "])

    def test_metadata_uses_configured_dimension_without_encoding(self) -> None:
        model = FakeModel()
        adapter = ClipViTB32Adapter(model=model, dimension=4)

        metadata = adapter.metadata()

        self.assertEqual(metadata.dimension, 4)
        self.assertEqual(model.batch_sizes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
