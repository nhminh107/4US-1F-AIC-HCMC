from __future__ import annotations

import unittest

from BackEnd.app.contracts.embedding import ClipRecord
from BackEnd.app.embedding.clip.planner import dedup_ratio, plan_video_work


def make_clip(clip_id: str, start_ms: int, end_ms: int) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        video_id="L21_V001",
        shot_id="shot-1",
        start_ms=start_ms,
        end_ms=end_ms,
        scale_type="fixed_window",
        target_num_frames=4,
        sampling_strategy="uniform_midpoint",
        sampling_version="clip-sampling@1.0.0",
        clip_builder_version="clip-builder@1.0.0",
    )


class WorkPlannerTests(unittest.TestCase):
    def test_groups_by_video_and_deduplicates_timestamps(self) -> None:
        clip_a = make_clip("a", 0, 1000)
        clip_b = make_clip("b", 0, 1000)

        work_units = plan_video_work([clip_b, clip_a])

        self.assertEqual(len(work_units), 1)
        self.assertEqual(work_units[0].unique_timestamps_ms, (125, 375, 625, 875))
        self.assertGreater(dedup_ratio(work_units[0]), 0)

    def test_max_clips_per_unit_chunks_work_units(self) -> None:
        clip_a = make_clip("a", 0, 1000)
        clip_b = make_clip("b", 1000, 2000)
        clip_c = make_clip("c", 2000, 3000)

        work_units = plan_video_work([clip_a, clip_b, clip_c], max_clips_per_unit=2)

        self.assertEqual(len(work_units), 2)
        self.assertEqual(len(work_units[0].sorted_clip_records), 2)
        self.assertEqual(len(work_units[1].sorted_clip_records), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

