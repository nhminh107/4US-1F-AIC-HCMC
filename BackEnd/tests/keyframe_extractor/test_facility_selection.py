"""Tests for semantic coverage selection used by the hybrid keyframe strategy."""

from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.keyframe_extractor.facility_selection import select_facility_representatives


class FacilitySelectionTests(unittest.TestCase):
    def test_prefers_semantic_coverage_over_duplicate_candidates(self) -> None:
        selected = select_facility_representatives(
            [10, 20, 30, 40],
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                    [0.01, 0.99],
                ],
                dtype=np.float32,
            ),
            max_representatives=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertTrue(any(index in selected for index in (10, 20)))
        self.assertTrue(any(index in selected for index in (30, 40)))

    def test_existing_frame_seeds_coverage(self) -> None:
        selected = select_facility_representatives(
            [10, 20, 30],
            np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32),
            reference_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            max_representatives=2,
            min_marginal_gain=0.02,
        )

        self.assertEqual(selected, [30])

    def test_does_not_add_a_candidate_already_covered_by_official_frame(self) -> None:
        selected = select_facility_representatives(
            [10, 20],
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            reference_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            max_representatives=2,
        )

        self.assertEqual(selected, [])

    def test_is_deterministic_when_gains_tie(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        self.assertEqual(
            select_facility_representatives([10, 20], vectors, max_representatives=1),
            [10],
        )


if __name__ == "__main__":
    unittest.main()
