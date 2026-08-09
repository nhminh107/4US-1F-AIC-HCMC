"""Simple integration test: one MP4 video -> Shots -> Clips.

Run directly from the project root:

    python BackEnd/tests/test_video_to_shots_to_clips.py data/video/L23_V001.mp4

The script prints every Shot and its logical Clip windows. It does not create
new MP4 files because ``ClipExtractor`` materializes metadata by default.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.clip_extractor import ClipExtractor
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.shot_extractor import ShotExtractor
from BackEnd.app.shot_extractor.shot_extractor import DEFAULT_WEIGHTS_PATH


PipelineResult = list[tuple[ShotMetadata, list[dict[str, Any]]]]


def run_video_to_shots_to_clips(
    video_path: Path,
    *,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    device: str | None = None,
) -> PipelineResult:
    """Run the two public extractor modules and validate their basic contracts."""

    video_path = video_path.expanduser().resolve()
    weights_path = weights_path.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("ShotExtractor currently expects an MP4 input")
    if not weights_path.is_file():
        raise FileNotFoundError(f"TransNetV2 weights do not exist: {weights_path}")

    shot_extractor = ShotExtractor(
        video_dir=video_path.parent,
        weights_path=weights_path,
        device=device,
    )
    clip_extractor = ClipExtractor()

    shots = shot_extractor.extract(video_path.stem)
    if not shots:
        raise AssertionError("ShotExtractor returned no Shots")

    result: PipelineResult = []
    for shot in shots:
        clips = clip_extractor.run(shot)
        _validate_clips(shot, clips)
        result.append((shot, clips))

    _print_result(video_path, result)
    return result


def _validate_clips(shot: ShotMetadata, clips: list[dict[str, Any]]) -> None:
    """Check that ClipExtractor output stays inside and covers its parent Shot."""

    if not clips:
        raise AssertionError(f"{shot.shot_id} produced no Clips")
    if clips[0]["start_ms"] != shot.start_ms:
        raise AssertionError(f"{shot.shot_id}: first Clip does not start with Shot")
    if clips[-1]["end_ms"] != shot.end_ms:
        raise AssertionError(f"{shot.shot_id}: final Clip does not end with Shot")

    previous_end_ms = shot.start_ms
    for clip in clips:
        if clip["shot_id"] != shot.shot_id:
            raise AssertionError(f"{shot.shot_id}: Clip has the wrong shot_id")
        if not (shot.start_ms <= clip["start_ms"] < clip["end_ms"] <= shot.end_ms):
            raise AssertionError(f"{shot.shot_id}: Clip is outside its parent Shot")
        if clip["start_ms"] > previous_end_ms:
            raise AssertionError(f"{shot.shot_id}: uncovered gap between Clips")
        previous_end_ms = max(previous_end_ms, clip["end_ms"])


def _print_result(video_path: Path, result: PipelineResult) -> None:
    """Print a compact hierarchy that is easy to inspect manually."""

    clip_count = sum(len(clips) for _, clips in result)
    print("\nVIDEO -> SHOTS -> CLIPS")
    print(f"Video : {video_path}")
    print(f"Shots : {len(result)}")
    print(f"Clips : {clip_count}\n")

    for shot, clips in result:
        print(
            f"SHOT {shot.shot_index:03d} | {shot.shot_id} | "
            f"{_seconds(shot.start_ms)} - {_seconds(shot.end_ms)} | "
            f"{len(clips)} clip(s)"
        )
        for clip in clips:
            print(
                f"  -> {clip['clip_id']} | "
                f"{_seconds(clip['start_ms'])} - {_seconds(clip['end_ms'])}"
            )


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:8.3f}s"


class VideoToShotsToClipsIntegrationTest(unittest.TestCase):
    """Allow the same integration check to run through unittest discovery."""

    def test_video_to_shots_to_clips(self) -> None:
        raw_video_path = os.environ.get("VIDEO_TO_CLIPS_TEST_VIDEO")
        if not raw_video_path:
            self.skipTest("set VIDEO_TO_CLIPS_TEST_VIDEO to a local MP4 path")

        result = run_video_to_shots_to_clips(Path(raw_video_path))

        self.assertGreater(len(result), 0)
        self.assertGreater(sum(len(clips) for _, clips in result), 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, help="Path to an input MP4 video")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to transnetv2-pytorch-weights.pth",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Inference device; default selects CUDA when available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_video_to_shots_to_clips(
        arguments.video_path,
        weights_path=arguments.weights_path,
        device=arguments.device,
    )
