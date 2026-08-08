"""Tests for HSV and CLIP redundancy elimination."""

from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.keyframe_extractor.redundancy import (
    RedundancyCandidate,
    eliminate_redundant_candidates,
)


def solid(color: tuple[int, int, int]) -> np.ndarray:
    return np.full((16, 16, 3), color, dtype=np.uint8)


def checker() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for row in range(16):
        for col in range(16):
            image[row, col] = (
                (row * 17) % 256,
                (col * 17) % 256,
                ((row + col) * 13) % 256,
            )
    return image


class RedundancyTests(unittest.TestCase):
    def test_filters_low_information_solid_frame(self) -> None:
        selected = eliminate_redundant_candidates(
            [
                RedundancyCandidate(10, solid((0, 0, 0)), [1.0, 0.0]),
                RedundancyCandidate(20, checker(), [0.0, 1.0]),
            ],
            max_output=5,
        )

        self.assertEqual(selected, [20])

    def test_keeps_similar_color_when_clip_vectors_differ(self) -> None:
        image = checker()
        selected = eliminate_redundant_candidates(
            [
                RedundancyCandidate(10, image, [1.0, 0.0], center_distance=0.1),
                RedundancyCandidate(20, image, [0.0, 1.0], center_distance=0.2),
            ],
            max_output=5,
            clip_similarity_threshold=0.95,
        )

        self.assertEqual(selected, [10, 20])

    def test_removes_duplicate_when_hsv_and_clip_agree(self) -> None:
        image = checker()
        selected = eliminate_redundant_candidates(
            [
                RedundancyCandidate(10, image, [1.0, 0.0], center_distance=0.1),
                RedundancyCandidate(20, image, [0.99, 0.01], center_distance=0.5),
                RedundancyCandidate(30, checker()[::-1], [0.0, 1.0], center_distance=0.2),
            ],
            max_output=5,
            clip_similarity_threshold=0.9,
        )

        self.assertEqual(selected, [10, 30])

    def test_caps_output_by_priority(self) -> None:
        selected = eliminate_redundant_candidates(
            [
                RedundancyCandidate(10, checker(), [1.0, 0.0], center_distance=0.3),
                RedundancyCandidate(20, checker()[::-1], [0.0, 1.0], center_distance=0.1),
            ],
            max_output=1,
        )

        self.assertEqual(selected, [20])


    def test_eliminate_cross_shot_duplicates(self) -> None:
        from BackEnd.app.keyframe_extractor.redundancy import eliminate_cross_shot_duplicates

        image = checker()
        shot1 = [RedundancyCandidate(10, image, [1.0, 0.0], center_distance=0.1)]
        shot2 = [
            RedundancyCandidate(20, image, [0.99, 0.01], center_distance=0.5),
            RedundancyCandidate(50, checker()[::-1], [0.0, 1.0], center_distance=0.2),
        ]

        cleaned = eliminate_cross_shot_duplicates(
            [shot1, shot2],
            hsv_similarity_threshold=0.75,
            clip_similarity_threshold=0.90,
            max_frame_gap=150,
        )

        # Shot 1 keeps [10], Shot 2 removes boundary duplicate [20] and keeps [50]
        self.assertEqual(cleaned, [[10], [50]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
