"""Tests for hybrid keyframe feature clustering."""

from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.keyframe_extractor.clustering import select_cluster_representatives


class ClusteringTests(unittest.TestCase):
    def test_empty_candidates_return_empty(self) -> None:
        selections = select_cluster_representatives([], np.empty((0, 2)), max_representatives=3)
        self.assertEqual(selections, [])

    def test_single_candidate_is_selected(self) -> None:
        selections = select_cluster_representatives([42], np.array([[1.0, 0.0]]), max_representatives=3)
        self.assertEqual([item.frame_idx for item in selections], [42])

    def test_selects_representatives_from_separated_clusters(self) -> None:
        frame_indices = [0, 10, 20, 100, 110, 120]
        vectors = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
                [5.2, 5.0],
            ],
            dtype=np.float32,
        )

        selections = select_cluster_representatives(
            frame_indices,
            vectors,
            max_representatives=3,
        )

        selected = [item.frame_idx for item in selections]
        self.assertIn(10, selected)
        self.assertIn(110, selected)
        self.assertEqual(selected, sorted(selected))

    def test_invalid_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_cluster_representatives([1, 2], np.array([[1.0, 0.0]]), max_representatives=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
