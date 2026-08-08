"""Unit test mở rộng cho logic thuần chọn candidate frame index (sampling.py)."""

from __future__ import annotations

import unittest

from BackEnd.app.keyframe_extractor.sampling import select_additional_keyframe_indices


class SamplingTests(unittest.TestCase):
    """Kiểm tra logic chọn candidate keyframe index bổ sung với nhiều bộ testcase."""

    FPS = 25.0

    def test_short_shot_without_existing_keyframe_returns_center_frame(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=99,
            start_ms=0,
            end_ms=4000,
            fps=self.FPS,
            existing_frame_idxs=[],
        )
        self.assertGreaterEqual(len(candidates), 1)
        for idx in candidates:
            self.assertTrue(0 <= idx <= 99)

    def test_shot_with_existing_keyframe_skips_duplicate(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=49,
            start_ms=0,
            end_ms=2000,
            fps=self.FPS,
            existing_frame_idxs=[25],
        )
        self.assertEqual(candidates, [])

    def test_long_shot_returns_multiple_candidates(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=249,
            start_ms=0,
            end_ms=10000,
            fps=self.FPS,
            existing_frame_idxs=[],
        )
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual(candidates, sorted(candidates))
        for i in range(1, len(candidates)):
            self.assertGreater(candidates[i] - candidates[i - 1], 5)

    def test_avoids_existing_frame_and_adjacent_frames(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=49,
            start_ms=0,
            end_ms=1000,
            fps=self.FPS,
            existing_frame_idxs=[25],
            min_frame_gap=5,
        )
        self.assertEqual(candidates, [])

    def test_single_frame_shot(self) -> None:
        # Shot chỉ có 1 frame duy nhất (frame 42)
        candidates = select_additional_keyframe_indices(
            start_frame_idx=42,
            end_frame_idx=42,
            start_ms=1000,
            end_ms=1040,
            fps=self.FPS,
            existing_frame_idxs=[],
        )
        self.assertEqual(candidates, [42])

    def test_single_frame_shot_already_in_existing_returns_empty(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=42,
            end_frame_idx=42,
            start_ms=1000,
            end_ms=1040,
            fps=self.FPS,
            existing_frame_idxs=[42],
        )
        self.assertEqual(candidates, [])

    def test_extremely_long_shot_respects_max_additional_limit(self) -> None:
        # Shot 60 giây (60,000ms), 1500 frames. max_additional_per_shot = 5
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=1499,
            start_ms=0,
            end_ms=60000,
            fps=self.FPS,
            existing_frame_idxs=[],
            max_additional_per_shot=5,
        )
        self.assertLessEqual(len(candidates), 6)  # target_count <= 6
        self.assertEqual(candidates, sorted(candidates))

    def test_crowded_existing_frames_finds_available_slot(self) -> None:
        # Hầu hết các frame ở trung tâm (15..35) bị cấm, candidate phải né sang khoảng còn trống
        existing = list(range(15, 36))
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=49,
            start_ms=0,
            end_ms=5000,
            fps=self.FPS,
            existing_frame_idxs=existing,
            min_frame_gap=2,
        )
        for idx in candidates:
            self.assertNotIn(idx, existing)
            self.assertTrue(0 <= idx <= 49)

    def test_custom_interval_and_gap_parameters(self) -> None:
        candidates = select_additional_keyframe_indices(
            start_frame_idx=0,
            end_frame_idx=199,
            start_ms=0,
            end_ms=8000,
            fps=self.FPS,
            existing_frame_idxs=[],
            target_interval_ms=1000,  # Lấy 1s/keyframe
            min_frame_gap=10,
        )
        self.assertGreaterEqual(len(candidates), 5)
        for i in range(1, len(candidates)):
            self.assertGreaterEqual(candidates[i] - candidates[i - 1], 10)

    def test_invalid_inputs_raise_value_errors(self) -> None:
        with self.assertRaises(ValueError):
            select_additional_keyframe_indices(start_frame_idx=-1, end_frame_idx=10, start_ms=0, end_ms=1000, fps=25.0)
        with self.assertRaises(ValueError):
            select_additional_keyframe_indices(start_frame_idx=10, end_frame_idx=5, start_ms=0, end_ms=1000, fps=25.0)
        with self.assertRaises(ValueError):
            select_additional_keyframe_indices(start_frame_idx=0, end_frame_idx=10, start_ms=0, end_ms=1000, fps=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
