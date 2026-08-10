"""Wiring tests bổ sung cho KeyframeExtractor với mock decode FFmpeg."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.config import HybridKeyframeConfig
from BackEnd.app.keyframe_extractor.hybrid_selector import HybridKeyframeSelectionError
from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor


class FakeHybridSelector:
    def __init__(self, selected=None, fail: bool = False) -> None:
        self.selected = list(selected or [])
        self.fail = fail
        self.calls = []

    def select(self, shot, *, video_path, fps, existing_frame_idxs=None):
        self.calls.append((shot, video_path, fps, existing_frame_idxs))
        if self.fail:
            raise HybridKeyframeSelectionError("fake hybrid failure")
        return list(self.selected)


class CountingHybridSelector:
    instances = []

    def __init__(self, config=None) -> None:
        self.config = config
        self.calls = []
        CountingHybridSelector.instances.append(self)

    def select(self, shot, *, video_path, fps, existing_frame_idxs=None):
        self.calls.append((shot, video_path, fps, existing_frame_idxs))
        return [shot.start_frame_idx + 1]


class ShotRelativeHybridSelector:
    def __init__(self) -> None:
        self.calls = []
        self.last_metrics = {"candidate_count": 1, "status": "success"}

    def select(self, shot, *, video_path, fps, existing_frame_idxs=None):
        self.calls.append((shot, video_path, fps, existing_frame_idxs))
        return [shot.start_frame_idx + 1]


class KeyframeExtractorWiringTests(unittest.TestCase):
    """Kiểm tra orchestrator của KeyframeExtractor với nhiều kịch bản testcase."""

    def test_extract_returns_valid_frame_metadata(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")

        shot = ShotMetadata(
            shot_id="L21_V001_S001",
            video_id="L21_V001",
            shot_index=1,
            start_ms=0,
            end_ms=10000,
            start_frame_idx=0,
            end_frame_idx=249,
        )

        fake_dimensions = [(1920, 1080)] * 5

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: fake_dimensions[: len(indices)],
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        for frame in frames:
            self.assertEqual(frame.video_id, "L21_V001")
            self.assertEqual(frame.shot_id, "L21_V001_S001")
            self.assertEqual(frame.frame_role, "keyframe")
            self.assertEqual(frame.source, "extracted")
            self.assertEqual(frame.width, 1920)
            self.assertEqual(frame.height, 1080)
            self.assertTrue(frame.frame_id.startswith("L21_V001_E"))
            self.assertLessEqual(len(frame.frame_id), 15, "frame_id phải vừa varchar(15)")

    def test_extract_for_video_sequences_seq_numbers(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")

        shots = [
            ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124),
            ShotMetadata("L21_V001_S002", "L21_V001", 1, 5000, 10000, 125, 249),
        ]

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames_chunked",
                side_effect=lambda v, indices, paths, chunk_size: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract_for_video("L21_V001", shots, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        seqs = [f.frame_id.split("_E")[1] for f in frames]
        self.assertEqual(seqs, [f"{i:03d}" for i in range(1, len(frames) + 1)])

    def test_extract_for_video_empty_shots_returns_empty(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        with patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0):
            frames = extractor.extract_for_video("L21_V001", [], existing_frame_idxs=[])
        self.assertEqual(frames, [])

    def test_extract_raises_error_if_start_frame_idx_is_none(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        shot_invalid = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, None, 124)
        with self.assertRaises(ValueError):
            extractor.extract(shot_invalid)

    def test_custom_seq_start_parameter(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        shot = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)
        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[], seq_start=10)
        self.assertTrue(frames[0].frame_id.endswith("_E010"))

    def test_hybrid_strategy_uses_injected_selector(self) -> None:
        selector = FakeHybridSelector(selected=[20, 90])
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip",
            hybrid_selector=selector,
        )
        shot = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)
        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[])

        self.assertEqual([frame.frame_idx for frame in frames], [20, 90])
        self.assertEqual(len(selector.calls), 1)

    def test_hybrid_strategy_falls_back_to_time_when_selector_fails(self) -> None:
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip",
            hybrid_selector=FakeHybridSelector(fail=True),
        )
        shot = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)
        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames",
                side_effect=lambda v, indices, paths: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract(shot, existing_frame_idxs=[])

        self.assertGreater(len(frames), 0)
        self.assertNotEqual([frame.frame_idx for frame in frames], [])

    def test_hybrid_strict_raises_when_selector_fails(self) -> None:
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip_strict",
            hybrid_config=HybridKeyframeConfig(fallback_to_time_sampling=False),
            hybrid_selector=FakeHybridSelector(fail=True),
        )
        shot = ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)
        with patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0):
            with self.assertRaises(HybridKeyframeSelectionError):
                extractor.extract(shot, existing_frame_idxs=[])

    def test_hybrid_selector_is_reused_across_video_shots(self) -> None:
        CountingHybridSelector.instances = []
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip",
        )
        shots = [
            ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124),
            ShotMetadata("L21_V001_S002", "L21_V001", 1, 5000, 10000, 125, 249),
            ShotMetadata("L21_V001_S003", "L21_V001", 2, 10000, 15000, 250, 374),
        ]
        with (
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.HybridKeyframeSelector",
                CountingHybridSelector,
            ),
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames_chunked",
                side_effect=lambda v, indices, paths, chunk_size: [(1920, 1080)] * len(indices),
            ),
        ):
            frames = extractor.extract_for_video("L21_V001", shots, existing_frame_idxs=[])

        self.assertEqual(len(frames), 3)
        self.assertEqual(len(CountingHybridSelector.instances), 1)
        self.assertEqual(len(CountingHybridSelector.instances[0].calls), 3)

    def test_extract_for_video_reports_progress_per_shot(self) -> None:
        selector = FakeHybridSelector(selected=[20])
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip",
            hybrid_selector=selector,
        )
        selector.last_metrics = {"candidate_count": 2, "decode_s": 0.25, "status": "success"}
        shots = [
            ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124),
            ShotMetadata("L21_V001_S002", "L21_V001", 1, 5000, 10000, 125, 249),
        ]
        events = []
        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames_chunked",
                side_effect=lambda v, indices, paths, chunk_size: [(1920, 1080)] * len(indices),
            ),
        ):
            extractor.extract_for_video(
                "L21_V001",
                shots,
                existing_frame_idxs=[],
                progress_callback=events.append,
            )

        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["shot_number"], 1)
        self.assertEqual(events[0]["total_shots"], 2)
        self.assertEqual(events[0]["phase"], "start")
        self.assertEqual(events[0]["selected_count"], 0)
        self.assertEqual(events[0]["strategy"], "hybrid_clip")
        self.assertEqual(events[1]["phase"], "done")
        self.assertEqual(events[1]["selected_count"], 1)
        self.assertEqual(events[1]["hybrid"]["candidate_count"], 2)
        self.assertEqual(events[-1]["phase"], "export")
        self.assertEqual(events[-1]["frame_count"], 2)

    def test_extract_for_video_exports_all_selected_frames_in_one_batch(self) -> None:
        selector = ShotRelativeHybridSelector()
        extractor = KeyframeExtractor(
            video_dir="data/video",
            keyframe_dir="data/keyframes",
            strategy="hybrid_clip",
            hybrid_selector=selector,
            max_frames_per_ffmpeg_batch=100,
        )
        shots = [
            ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124),
            ShotMetadata("L21_V001_S002", "L21_V001", 1, 5000, 10000, 125, 249),
            ShotMetadata("L21_V001_S003", "L21_V001", 2, 10000, 15000, 250, 374),
        ]

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch(
                "BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames_chunked",
                side_effect=lambda v, indices, paths, chunk_size: [(1920, 1080)] * len(indices),
            ) as export_mock,
        ):
            frames = extractor.extract_for_video("L21_V001", shots, existing_frame_idxs=[])

        self.assertEqual([frame.frame_idx for frame in frames], [1, 126, 251])
        self.assertEqual([frame.frame_id for frame in frames], ["L21_V001_E001", "L21_V001_E002", "L21_V001_E003"])
        self.assertEqual(export_mock.call_count, 1)
        self.assertEqual(list(export_mock.call_args.args[1]), [1, 126, 251])
        self.assertEqual(export_mock.call_args.kwargs["chunk_size"], 100)

    def test_extract_for_video_skips_export_when_no_candidates(self) -> None:
        extractor = KeyframeExtractor(video_dir="data/video", keyframe_dir="data/keyframes")
        shots = [ShotMetadata("L21_V001_S001", "L21_V001", 0, 0, 5000, 0, 124)]

        with (
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.probe_fps", return_value=25.0),
            patch.object(extractor, "_select_candidate_indices", return_value=[]),
            patch("BackEnd.app.keyframe_extractor.keyframe_extractor.extract_and_save_frames_chunked") as export_mock,
        ):
            frames = extractor.extract_for_video("L21_V001", shots, existing_frame_idxs=[])

        self.assertEqual(frames, [])
        export_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
