from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.embedding.evaluation.exact_retrieval import (
    exact_top_k,
    mean_reciprocal_rank,
    recall_at_k,
)
from BackEnd.app.embedding.evaluation.text_encoder import encode_clip_queries


class ExactRetrievalTests(unittest.TestCase):
    def test_exact_top_k_returns_sorted_inner_product_results(self) -> None:
        queries = np.array([[1, 0]], dtype=np.float32)
        visuals = np.array([[0, 1], [1, 0], [0.5, 0.5]], dtype=np.float32)

        scores, indexes = exact_top_k(queries, visuals, top_k=2)

        self.assertEqual(indexes.tolist(), [[1, 2]])
        self.assertGreater(scores[0, 0], scores[0, 1])

    def test_recall_and_mrr_use_labeled_relevant_indexes(self) -> None:
        indexes = np.array([[4, 2, 1], [0, 3, 5]], dtype=np.int64)

        self.assertEqual(recall_at_k(indexes, [{2}, {9}], 2), 0.5)
        self.assertEqual(mean_reciprocal_rank(indexes, [{2}, {3}]), 0.5)

    def test_text_encoder_uses_injected_adapter(self) -> None:
        class FakeAdapter:
            def encode_texts(self, texts):
                return np.ones((len(texts), 2), dtype=np.float32)

        vectors = encode_clip_queries(["a", "b"], adapter=FakeAdapter())

        self.assertEqual(vectors.shape, (2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

