from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.contracts.embedding import ClipRecord
from BackEnd.app.embedding.clip.aggregator import aggregate_clip_frames
from BackEnd.app.embedding.common.quality import validate_embedding_matrix
from BackEnd.app.embedding.shot.aggregator import aggregate_shot_clips
from BackEnd.app.embedding.shot.coverage import coverage_weights


def make_clip(clip_id: str, start_ms: int, end_ms: int) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        video_id="L21_V001",
        shot_id="shot-1",
        start_ms=start_ms,
        end_ms=end_ms,
        scale_type="fixed_window",
        target_num_frames=16,
        sampling_strategy="uniform_midpoint",
        sampling_version="clip-sampling@1.0.0",
        clip_builder_version="clip-builder@1.0.0",
    )


class ClipAggregationTests(unittest.TestCase):
    def test_masked_mean_ignores_invalid_vectors(self) -> None:
        vectors = np.array([[1, 0], [0, 1], [100, 100]], dtype=np.float32)
        result = aggregate_clip_frames(vectors, [True, True, False])

        expected = np.array([1, 1], dtype=np.float32) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_masked_mean_rejects_all_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            aggregate_clip_frames(np.ones((2, 4), dtype=np.float32), [False, False])

    def test_quality_gate_rejects_wrong_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_embedding_matrix(np.ones((1, 3), dtype=np.float32), dimension=2)


class ShotAggregationTests(unittest.TestCase):
    def test_coverage_weights_split_overlap_once(self) -> None:
        clips = [make_clip("a", 0, 10), make_clip("b", 5, 15)]

        self.assertEqual(coverage_weights(clips), [7.5, 7.5])

    def test_shot_aggregation_uses_duration_weights_without_overlap(self) -> None:
        clips = [make_clip("a", 0, 10), make_clip("b", 10, 30)]
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)

        result = aggregate_shot_clips(clips, vectors)

        expected = np.array([10, 20], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shot_aggregation_rejects_empty_clip_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one clip"):
            aggregate_shot_clips([], np.empty((0, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
