"""Tests for sparse hybrid keyframe candidate sampling."""

from __future__ import annotations

import unittest

from BackEnd.app.keyframe_extractor.candidate_sampler import sample_candidate_frame_indices


class CandidateSamplerTests(unittest.TestCase):
    def test_samples_sparse_frames_inside_shot_bounds(self) -> None:
        candidates = sample_candidate_frame_indices(
            start_frame_idx=0,
            end_frame_idx=249,
            fps=25.0,
            sample_fps=1.0,
            max_candidate_frames_per_shot=64,
        )

        self.assertEqual(candidates, sorted(candidates))
        self.assertTrue(all(0 <= frame_idx <= 249 for frame_idx in candidates))
        self.assertGreaterEqual(len(candidates), 9)
        self.assertLessEqual(len(candidates), 12)

    def test_respects_existing_frames_and_min_gap(self) -> None:
        candidates = sample_candidate_frame_indices(
            start_frame_idx=0,
            end_frame_idx=99,
            fps=25.0,
            existing_frame_idxs=[50],
            sample_fps=1.0,
            min_frame_gap=5,
        )

        self.assertTrue(all(abs(frame_idx - 50) > 5 for frame_idx in candidates))

    def test_caps_long_shot_candidates_evenly(self) -> None:
        candidates = sample_candidate_frame_indices(
            start_frame_idx=0,
            end_frame_idx=2999,
            fps=25.0,
            sample_fps=2.0,
            max_candidate_frames_per_shot=16,
        )

        self.assertLessEqual(len(candidates), 16)
        self.assertEqual(candidates, sorted(candidates))
        self.assertGreater(candidates[-1], candidates[0])

    def test_short_crowded_shot_returns_nearest_allowed_candidate(self) -> None:
        candidates = sample_candidate_frame_indices(
            start_frame_idx=10,
            end_frame_idx=12,
            fps=25.0,
            existing_frame_idxs=[11],
            min_frame_gap=0,
        )

        self.assertEqual(candidates, [10])

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            sample_candidate_frame_indices(start_frame_idx=5, end_frame_idx=4, fps=25.0)
        with self.assertRaises(ValueError):
            sample_candidate_frame_indices(start_frame_idx=0, end_frame_idx=4, fps=0)
        with self.assertRaises(ValueError):
            sample_candidate_frame_indices(start_frame_idx=0, end_frame_idx=4, fps=25.0, sample_fps=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
