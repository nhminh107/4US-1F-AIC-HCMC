"""Test ``sampling.sample_frames_in_range`` — logic thuần, không cần VLM/GPU.

Xem ``Markdown_Doc/module_caption.md`` mục 3.2 (hướng Multi-frame sampling)
và ``BackEnd/app/caption/sampling.py`` cho ngữ cảnh đầy đủ.
"""

from __future__ import annotations

import unittest

from BackEnd.app.caption.sampling import sample_frames_in_range
from BackEnd.app.contracts.pipeline import ClipWindowMetadata, FrameMetadata


def _frame(frame_id: str, timestamp_ms: int) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V001",
        shot_id="L21_V001_S000",
        timestamp_ms=timestamp_ms,
        fps=25.0,
        frame_idx=timestamp_ms // 40,
        source="official",
    )


def _clip(start_ms: int, end_ms: int) -> ClipWindowMetadata:
    return ClipWindowMetadata(clip_id="L21_V001_C000", shot_id="L21_V001_S000", start_ms=start_ms, end_ms=end_ms)


class SampleFramesInRangeTests(unittest.TestCase):
    def test_filters_by_half_open_interval(self) -> None:
        clip = _clip(start_ms=1000, end_ms=2000)
        frames = [_frame("f0", 999), _frame("f1", 1000), _frame("f2", 1500), _frame("f3", 2000), _frame("f4", 2001)]

        result = sample_frames_in_range(clip, frames, num_samples=10)

        # end_ms không được bao gồm (nửa-mở), start_ms được bao gồm.
        self.assertEqual([f.frame_id for f in result], ["f1", "f2"])

    def test_returns_all_when_fewer_candidates_than_num_samples(self) -> None:
        clip = _clip(start_ms=0, end_ms=1000)
        frames = [_frame("f0", 100), _frame("f1", 500)]

        result = sample_frames_in_range(clip, frames, num_samples=6)

        self.assertEqual([f.frame_id for f in result], ["f0", "f1"])

    def test_downsamples_uniformly_when_more_candidates_than_num_samples(self) -> None:
        clip = _clip(start_ms=0, end_ms=1000)
        frames = [_frame(f"f{i}", i * 100) for i in range(10)]  # f0..f9, mỗi 100ms

        result = sample_frames_in_range(clip, frames, num_samples=4)

        self.assertEqual(len(result), 4)
        # Kết quả phải giữ đúng thứ tự thời gian, không trùng frame.
        timestamps = [f.timestamp_ms for f in result]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(set(f.frame_id for f in result)), len(result))
        # Frame đầu và cuối trong khoảng phải luôn được giữ lại (linspace bao 2 đầu mút).
        self.assertEqual(result[0].frame_id, "f0")
        self.assertEqual(result[-1].frame_id, "f9")

    def test_returns_empty_list_when_no_frame_in_range(self) -> None:
        clip = _clip(start_ms=5000, end_ms=6000)
        frames = [_frame("f0", 100), _frame("f1", 200)]

        result = sample_frames_in_range(clip, frames, num_samples=4)

        self.assertEqual(result, [])

    def test_result_sorted_even_if_input_unordered(self) -> None:
        clip = _clip(start_ms=0, end_ms=1000)
        frames = [_frame("f2", 500), _frame("f0", 100), _frame("f1", 300)]

        result = sample_frames_in_range(clip, frames, num_samples=10)

        self.assertEqual([f.frame_id for f in result], ["f0", "f1", "f2"])

    def test_raises_on_non_positive_num_samples(self) -> None:
        clip = _clip(start_ms=0, end_ms=1000)
        with self.assertRaises(ValueError):
            sample_frames_in_range(clip, [], num_samples=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
