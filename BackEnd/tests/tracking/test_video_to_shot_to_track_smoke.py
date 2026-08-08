"""Opt-in smoke test: one local video -> shots -> ByteTrack results."""

from __future__ import annotations

import os
import unittest
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from BackEnd.app.contracts.pipeline import (
    ObjectDetectionResult,
    ObjectTrackResult,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data/video/L21_V001.mp4"
RUN_SMOKE_TEST = os.environ.get("RUN_VIDEO_TO_TRACKING_SMOKE") == "1"


@unittest.skipUnless(
    RUN_SMOKE_TEST,
    "Set RUN_VIDEO_TO_TRACKING_SMOKE=1 to run real video inference.",
)
class VideoToShotToTrackSmokeTests(unittest.TestCase):
    """Runs the real ShotExtractor and detector on one local video."""

    def test_video_to_shot_to_track(self) -> None:
        video_path = Path(
            os.environ.get("VIDEO_TO_TRACKING_PATH", str(DEFAULT_VIDEO_PATH))
        ).resolve()
        self.assertTrue(video_path.is_file(), f"Video not found: {video_path}")

        # Heavy imports stay inside the opt-in test so normal unit-test discovery is fast.
        from BackEnd.app.shot_extractor import ShotExtractor
        from BackEnd.app.tracking import ByteTrackService, TrackingConfig

        video = VideoMetadata(video_id=video_path.stem, video_path=video_path)
        shots = ShotExtractor().extract(video.video_id)
        self.assertGreater(len(shots), 0, "ShotExtractor returned no shots.")

        max_shots = int(os.environ.get("VIDEO_TO_TRACKING_MAX_SHOTS", "1"))
        selected_shots = shots[:max_shots] if max_shots > 0 else shots
        sampling_fps = float(os.environ.get("VIDEO_TO_TRACKING_SAMPLING_FPS", "1"))
        result = ByteTrackService(
            config=TrackingConfig(sampling_fps=sampling_fps),
        ).track_video(video, selected_shots)

        self.assertTrue(all(isinstance(item, ShotMetadata) for item in selected_shots))
        self.assertTrue(
            all(isinstance(item, ObjectDetectionResult) for item in result.detections)
        )
        self.assertTrue(all(isinstance(item, ObjectTrackResult) for item in result.tracks))
        self.assertTrue(
            all(isinstance(item, TrackObservationResult) for item in result.observations)
        )
        self.assertTrue(
            {track.shot_id for track in result.tracks}.issubset(
                {shot.shot_id for shot in selected_shots}
            )
        )

        print("\n=== ShotMetadata ===")
        pprint([asdict(shot) for shot in selected_shots])
        print("\n=== ObjectDetectionResult (first 10) ===")
        pprint([asdict(item) for item in result.detections[:10]])
        print("\n=== ObjectTrackResult ===")
        pprint([asdict(track) for track in result.tracks])
        print("\n=== TrackObservationResult (first 20) ===")
        pprint([asdict(item) for item in result.observations[:20]])
        print(
            "\nSummary: "
            f"shots={len(selected_shots)}, detections={len(result.detections)}, "
            f"tracks={len(result.tracks)}, observations={len(result.observations)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
