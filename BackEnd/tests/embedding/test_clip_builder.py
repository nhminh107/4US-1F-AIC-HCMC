from __future__ import annotations

import unittest

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding.clip.builder import build_clips


class ClipBuilderTests(unittest.TestCase):
    def test_short_shot_creates_one_full_shot_clip(self) -> None:
        clips = build_clips(
            [ShotMetadata("shot-1", "L21_V001", 0, 1000, 4000)]
        )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].scale_type, "full_shot")
        self.assertEqual((clips[0].start_ms, clips[0].end_ms), (1000, 4000))

    def test_exact_ten_second_shot_is_full_shot(self) -> None:
        clips = build_clips([ShotMetadata("shot-1", "L21_V001", 0, 0, 10_000)])

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].scale_type, "full_shot")

    def test_slightly_long_shot_aligns_tail_without_duplicate(self) -> None:
        clips = build_clips([ShotMetadata("shot-1", "L21_V001", 0, 0, 10_001)])

        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0].start_ms, clips[0].end_ms), (1, 10_001))

    def test_twenty_five_second_shot_covers_tail(self) -> None:
        clips = build_clips([ShotMetadata("shot-1", "L21_V001", 0, 0, 25_000)])
        intervals = [(clip.start_ms, clip.end_ms) for clip in clips]

        self.assertEqual(intervals, [(0, 10_000), (8_000, 18_000), (15_000, 25_000)])
        self.assertEqual(clips[-1].end_ms, 25_000)

    def test_duplicate_shot_input_is_ignored(self) -> None:
        shot = ShotMetadata("shot-1", "L21_V001", 0, 0, 4000)
        clips = build_clips([shot, shot])

        self.assertEqual(len(clips), 1)

    def test_clip_id_is_deterministic(self) -> None:
        shot = ShotMetadata("shot-1", "L21_V001", 0, 0, 4000)

        self.assertEqual(build_clips([shot])[0].clip_id, build_clips([shot])[0].clip_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

