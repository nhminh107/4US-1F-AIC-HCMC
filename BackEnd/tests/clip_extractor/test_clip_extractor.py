from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from BackEnd.app.clip_extractor import (
    ClipExtractor,
    ClipExtractorConfig,
    InvalidShotError,
)


class ClipExtractorMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ClipExtractor()

    def test_short_shot_creates_one_full_shot_clip(self) -> None:
        clips = self.extractor.run(
            {
                "shot_id": "L21_V001_S0001",
                "video_id": "L21_V001",
                "start_ms": 2_000,
                "end_ms": 12_000,
                "start_frame_idx": 60,
                "end_frame_idx": 360,
            }
        )
        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start_ms"], clips[0]["end_ms"]), (2_000, 12_000))

    def test_long_shot_uses_overlapping_fixed_windows(self) -> None:
        clips = self.extractor.run(
            {
                "shot_id": "L21_V001_S0003",
                "video_id": "L21_V001",
                "start_ms": 80_000,
                "end_ms": 105_000,
                "start_frame_idx": 2_400,
                "end_frame_idx": 3_150,
            }
        )

        self.assertEqual(len(clips), 3)
        self.assertTrue(all(isinstance(clip, dict) for clip in clips))
        self.assertEqual(clips[0]["start_ms"], 80_000)
        self.assertEqual(clips[-1]["end_ms"], 105_000)
        self.assertEqual(
            [(clip["start_ms"], clip["end_ms"]) for clip in clips],
            [(80_000, 90_000), (88_000, 98_000), (95_000, 105_000)],
        )
        self.assertTrue(
            all(clip["end_ms"] - clip["start_ms"] <= 10_000 for clip in clips)
        )
        self.assertEqual(clips[0]["clip_id"], "L21V001S0003C01")
        self.assertEqual(
            [
                (clip["start_frame_idx"], clip["end_frame_idx"])
                for clip in clips
            ],
            [(2_400, 2_700), (2_640, 2_940), (2_850, 3_150)],
        )
        self.assertTrue(all(clip["sampling_fps"] == 30.0 for clip in clips))
        self.assertTrue(all(len(clip["clip_id"]) <= 15 for clip in clips))

        for left, right in zip(clips, clips[1:]):
            self.assertLess(right["start_ms"], left["end_ms"])
            self.assertLess(right["start_frame_idx"], left["end_frame_idx"])

    def test_shot_slightly_over_threshold_does_not_create_tiny_tail(self) -> None:
        clips = self.extractor.run(
            {
                "shot_id": "shot-1",
                "video_id": "video-1",
                "start_ms": 0,
                "end_ms": 10_001,
                "start_frame_idx": 0,
                "end_frame_idx": 300,
            }
        )
        self.assertEqual(len(clips), 2)
        self.assertEqual(
            [(clip["start_ms"], clip["end_ms"]) for clip in clips],
            [(0, 10_000), (1, 10_001)],
        )

    def test_run_returns_list_of_metadata_objects(self) -> None:
        clips = self.extractor.run(
            {
            "shot_id": "shot-1",
            "video_id": "video-1",
            "start_ms": 0,
            "end_ms": 20_000,
            "start_frame_idx": 0,
            "end_frame_idx": 600,
            }
        )

        # In toàn bộ list metadata ra Terminal
        print("\nOUTPUT LIST:")
        print(json.dumps(clips, ensure_ascii=False, indent=2))

        self.assertIsInstance(clips, list)
        self.assertTrue(all(isinstance(clip, dict) for clip in clips))

        clip = clips[0]

        self.assertEqual(
        set(clip),
        {
            "clip_id",
            "shot_id",
            "start_ms",
            "end_ms",
            "start_frame_idx",
            "end_frame_idx",
            "sampling_fps",
            "clip_path",
        },
    )
        self.assertIsNone(clip["clip_path"])
    def test_invalid_shot_is_rejected(self) -> None:
        with self.assertRaises(InvalidShotError):
            self.extractor.run(
                {
                    "shot_id": "shot-1",
                    "video_id": "video-1",
                    "start_ms": 5_000,
                    "end_ms": 5_000,
                    "start_frame_idx": 150,
                    "end_frame_idx": 150,
                }
            )

    def test_custom_clip_id_factory(self) -> None:
        extractor = ClipExtractor(
            clip_id_factory=lambda shot, index: "%s-part-%d" % (shot.video_id, index)
        )
        clips = extractor.run(
            {
                "shot_id": "shot-1",
                "video_id": "video-1",
                "start_ms": 0,
                "end_ms": 12_000,
                "start_frame_idx": 0,
                "end_frame_idx": 360,
            }
        )
        self.assertEqual(
            [clip["clip_id"] for clip in clips],
            ["video-1-part-1", "video-1-part-2"],
        )

        with self.assertRaises(InvalidShotError):
            ClipExtractor(clip_id_factory=lambda shot, index: "X" * 16).run(
                {
                    "shot_id": "shot-1",
                    "video_id": "video-1",
                    "start_ms": 0,
                    "end_ms": 12_000,
                    "start_frame_idx": 0,
                    "end_frame_idx": 360,
                }
            )


@unittest.skipUnless(
    shutil.which("ffmpeg")
    and shutil.which("ffprobe")
    and os.environ.get("CLIP_EXTRACTOR_TEST_VIDEO"),
    "Real-video test requires FFmpeg and CLIP_EXTRACTOR_TEST_VIDEO",
)
class ClipExtractorFFmpegTests(unittest.TestCase):
    def test_materialize_real_dataset_video(self) -> None:
        source_path = Path(os.environ["CLIP_EXTRACTOR_TEST_VIDEO"]).resolve()
        self.assertTrue(source_path.is_file(), "Video does not exist: %s" % source_path)

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=avg_frame_rate",
                "-of",
                "json",
                str(source_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        media_info = json.loads(probe.stdout)
        source_duration_ms = round(
            float(media_info["format"]["duration"]) * 1000
        )
        source_fps = float(Fraction(media_info["streams"][0]["avg_frame_rate"]))

        self.assertGreater(source_duration_ms, 10_100)
        self.assertGreater(source_fps, 0)

        test_end_ms = min(25_000, source_duration_ms - 100)
        test_end_frame_idx = max(1, round(test_end_ms * source_fps / 1000))
        video_id = source_path.stem
        self.assertLessEqual(len(video_id), 15)
        shot_id = "%s_S0001" % video_id
        if len(shot_id) > 15:
            shot_id = "REAL_S0001"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            output_root = temp_path / "clips"

            extractor = ClipExtractor(
                ClipExtractorConfig(
                    materialize_files=True,
                    output_root=output_root,
                    overwrite=True,
                    validate_source_duration=True,
                    preset="ultrafast",
                )
            )
            clips = extractor.run(
                {
                    "shot_id": shot_id,
                    "video_id": video_id,
                    "start_ms": 0,
                    "end_ms": test_end_ms,
                    "start_frame_idx": 0,
                    "end_frame_idx": test_end_frame_idx,
                },
                video_path=source_path,
            )

            self.assertGreaterEqual(len(clips), 2)
            self.assertTrue(all(isinstance(clip, dict) for clip in clips))
            self.assertEqual(clips[0]["start_ms"], 0)
            self.assertEqual(clips[-1]["end_ms"], test_end_ms)
            self.assertEqual(clips[0]["start_frame_idx"], 0)
            self.assertEqual(clips[-1]["end_frame_idx"], test_end_frame_idx)

            for left, right in zip(clips, clips[1:]):
                self.assertLessEqual(right["start_ms"], left["end_ms"])
                self.assertLessEqual(right["start_frame_idx"], left["end_frame_idx"])

            for clip in clips:
                duration_ms = clip["end_ms"] - clip["start_ms"]
                clip_path = Path(clip["clip_path"])
                self.assertLessEqual(len(clip["clip_id"]), 15)
                self.assertLessEqual(duration_ms, 10_000)
                self.assertTrue(clip_path.is_file())
                self.assertGreater(clip_path.stat().st_size, 0)
                self.assertLessEqual(len(str(clip_path)), 200)

                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        str(clip_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                actual_duration_ms = round(
                    float(json.loads(probe.stdout)["format"]["duration"]) * 1000
                )
                self.assertLessEqual(abs(actual_duration_ms - duration_ms), 150)


if __name__ == "__main__":
    unittest.main()
