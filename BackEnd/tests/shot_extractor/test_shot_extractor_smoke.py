"""Smoke test end-to-end cho ShotExtractor, chạy trên một video local thật.

Theo ``agent.md`` mục 12, các lần chạy với dữ liệu thật/lớn là nấc cuối cùng
của thang test và chỉ có ý nghĩa "khi thực sự cần" — vì vậy test này tự
``skip`` (không phải ``fail``) khi thiếu một điều kiện tiên quyết mà nó
không kiểm soát được trên máy đang chạy:

- ``ffmpeg``/``ffprobe`` có trong PATH (xem ``video_decoder.py``);
- trọng số TransNetV2 đã convert (xem ``shot_extractor.py``);
- có ít nhất một video mẫu local trong ``data/video/`` thuộc tập L21/L23
  nêu ở mục 0 của ``Markdown_Doc/module_shot_keyframe.md``.

Nhờ vậy `python -m unittest` vẫn chạy sạch trên một checkout mới, nhưng vẫn
cho kiểm tra đúng đắn thật sự trên máy đã cấu hình đầy đủ (máy có GPU, đã
cài FFmpeg và convert xong weights).
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from BackEnd.app.shot_extractor.shot_extractor import DEFAULT_VIDEO_DIR, DEFAULT_WEIGHTS_PATH, ShotExtractor

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_WEIGHTS_AVAILABLE = DEFAULT_WEIGHTS_PATH.is_file()


def _first_sample_video_id() -> str | None:
    if not DEFAULT_VIDEO_DIR.is_dir():
        return None
    videos = sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    return videos[0].stem if videos else None


_SAMPLE_VIDEO_ID = _first_sample_video_id()


@unittest.skipUnless(_FFMPEG_AVAILABLE, "ffmpeg/ffprobe not found on PATH")
@unittest.skipUnless(_WEIGHTS_AVAILABLE, f"TransNetV2 weights not found at {DEFAULT_WEIGHTS_PATH}")
@unittest.skipUnless(_SAMPLE_VIDEO_ID is not None, f"no sample .mp4 under {DEFAULT_VIDEO_DIR}")
class ShotExtractorSmokeTests(unittest.TestCase):
    """Chạy pipeline TransNetV2 thật, end-to-end, trên một video local."""

    def test_extract_returns_ordered_non_overlapping_shots(self) -> None:
        extractor = ShotExtractor()
        shots = extractor.extract(_SAMPLE_VIDEO_ID)

        self.assertGreater(len(shots), 0, "expected at least one shot")

        previous_end_ms = -1
        previous_end_frame_idx = -1
        for index, shot in enumerate(shots):
            self.assertEqual(shot.video_id, _SAMPLE_VIDEO_ID)
            self.assertEqual(shot.shot_index, index)
            self.assertEqual(shot.shot_id, f"{_SAMPLE_VIDEO_ID}_S{index:03d}")
            self.assertLessEqual(len(shot.shot_id), 15, "shot_id must fit varchar(15)")
            self.assertGreater(shot.end_ms, shot.start_ms)
            self.assertGreaterEqual(shot.start_ms, previous_end_ms)
            self.assertIsNotNone(shot.start_frame_idx)
            self.assertIsNotNone(shot.end_frame_idx)
            self.assertGreater(shot.end_frame_idx, previous_end_frame_idx)
            previous_end_ms = shot.end_ms
            previous_end_frame_idx = shot.end_frame_idx


if __name__ == "__main__":
    unittest.main(verbosity=2)
