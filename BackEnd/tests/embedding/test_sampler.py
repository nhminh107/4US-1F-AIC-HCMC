from __future__ import annotations

import unittest

from BackEnd.app.contracts.embedding import ClipRecord
from BackEnd.app.embedding.clip.sampler import uniform_midpoint_timestamps


def make_clip(start_ms: int, end_ms: int, target_num_frames: int = 16) -> ClipRecord:
    return ClipRecord(
        clip_id="clip-1",
        video_id="L21_V001",
        shot_id="shot-1",
        start_ms=start_ms,
        end_ms=end_ms,
        scale_type="fixed_window",
        target_num_frames=target_num_frames,
        sampling_strategy="uniform_midpoint",
        sampling_version="clip-sampling@1.0.0",
        clip_builder_version="clip-builder@1.0.0",
    )


class SamplerTests(unittest.TestCase):
    def test_midpoint_formula_for_one_second_clip(self) -> None:
        timestamps = uniform_midpoint_timestamps(make_clip(0, 1000, 4))

        self.assertEqual(timestamps, (125, 375, 625, 875))

    def test_short_clip_timestamps_stay_inside_interval(self) -> None:
        timestamps = uniform_midpoint_timestamps(make_clip(100, 600, 16))

        self.assertEqual(len(timestamps), 16)
        self.assertTrue(all(100 <= timestamp < 600 for timestamp in timestamps))

    def test_supports_eight_sixteen_and_thirty_two_frames(self) -> None:
        clip = make_clip(0, 10_000)

        self.assertEqual(len(uniform_midpoint_timestamps(clip, 8)), 8)
        self.assertEqual(len(uniform_midpoint_timestamps(clip, 16)), 16)
        self.assertEqual(len(uniform_midpoint_timestamps(clip, 32)), 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)

