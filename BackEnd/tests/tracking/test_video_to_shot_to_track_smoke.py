"""Opt-in GPU smoke test for YOLO26 tracking on a short video range."""

from __future__ import annotations

import os
import unittest
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from BackEnd.app.contracts.pipeline import (
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
    """Runs real YOLO26 tracking without invoking unrelated pipeline models."""

    def test_video_to_shot_to_track(self) -> None:
        video_path = Path(
            os.environ.get("VIDEO_TO_TRACKING_PATH", str(DEFAULT_VIDEO_PATH))
        ).resolve()
        self.assertTrue(video_path.is_file(), f"Video not found: {video_path}")

        # Heavy imports stay inside the opt-in test so unit-test discovery stays fast.
        from BackEnd.app.tracking import ByteTrackService, TrackingConfig

        video = VideoMetadata(video_id=video_path.stem, video_path=video_path)
        duration_ms = int(os.environ.get("VIDEO_TO_TRACKING_DURATION_MS", "10000"))
        self.assertGreater(duration_ms, 0)
        selected_shots = [
            ShotMetadata(
                shot_id="tracking-smoke",
                video_id=video.video_id,
                shot_index=0,
                start_ms=0,
                end_ms=duration_ms,
            )
        ]
        sampling_fps = float(os.environ.get("VIDEO_TO_TRACKING_SAMPLING_FPS", "1"))
        model_path = Path(
            os.environ.get(
                "VIDEO_TO_TRACKING_MODEL_PATH",
                str(PROJECT_ROOT / "data/models/yolo26n.pt"),
            )
        )
        result = ByteTrackService(
            config=TrackingConfig(
                model_path=model_path,
                sampling_fps=sampling_fps,
                device=os.environ.get("VIDEO_TO_TRACKING_DEVICE", "cuda:0"),
            ),
        ).track_video(video, selected_shots)

        self.assertTrue(all(isinstance(item, ShotMetadata) for item in selected_shots))
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
        print("\n=== ObjectTrackResult ===")
        pprint([asdict(track) for track in result.tracks])
        print("\n=== TrackObservationResult (first 20) ===")
        pprint([asdict(item) for item in result.observations[:20]])
        print(
            "\nSummary: "
            f"shots={len(selected_shots)}, tracks={len(result.tracks)}, "
            f"observations={len(result.observations)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
