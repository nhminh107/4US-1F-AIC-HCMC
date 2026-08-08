"""Run shot detection and additional keyframe extraction for one local MP4 file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata
from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor
from BackEnd.app.shot_extractor.shot_extractor import DEFAULT_WEIGHTS_PATH, ShotExtractor

DEFAULT_RESULT_DIR = PROJECT_ROOT / "BackEnd" / "tests" / "result"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect shots and extract additional keyframes for one MP4 video. "
            "Images and metadata are saved under the output directory."
        )
    )
    parser.add_argument("video_path", type=Path, help="Path to the input .mp4 video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Directory for images and JSON output (default: {DEFAULT_RESULT_DIR}).",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to the converted TransNetV2 .pth checkpoint used for shot detection.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Inference device. Omit to use CUDA automatically when available.",
    )
    parser.add_argument(
        "--existing-frame-indices",
        default="",
        help="Optional comma-separated official frame indices to exclude, for example: 10,25,100.",
    )
    return parser.parse_args()


def _validate_video_path(video_path: Path) -> Path:
    resolved_path = video_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {resolved_path}")
    if resolved_path.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 input is supported by the extractors: {resolved_path}")
    return resolved_path


def _parse_existing_frame_indices(raw_indices: str) -> list[int]:
    if not raw_indices.strip():
        return []

    indices = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    if any(index < 0 for index in indices):
        raise ValueError("existing frame indices must be non-negative integers.")
    return indices


def _shot_to_dict(shot: ShotMetadata) -> dict[str, int | str | None]:
    return {
        "shot_id": shot.shot_id,
        "video_id": shot.video_id,
        "shot_index": shot.shot_index,
        "start_ms": shot.start_ms,
        "end_ms": shot.end_ms,
        "start_frame_idx": shot.start_frame_idx,
        "end_frame_idx": shot.end_frame_idx,
    }


def _frame_to_dict(frame: FrameMetadata) -> dict[str, int | float | str | None]:
    return {
        "frame_id": frame.frame_id,
        "video_id": frame.video_id,
        "shot_id": frame.shot_id,
        "timestamp_ms": frame.timestamp_ms,
        "fps": frame.fps,
        "frame_idx": frame.frame_idx,
        "frame_role": frame.frame_role,
        "source": frame.source,
        "n": frame.n,
        "pts_time": frame.pts_time,
        "frame_path": str(frame.frame_path) if frame.frame_path is not None else None,
        "width": frame.width,
        "height": frame.height,
    }


def main() -> None:
    args = _parse_args()
    video_path = _validate_video_path(args.video_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_extractor = ShotExtractor(
        weights_path=args.weights_path.expanduser().resolve(),
        device=args.device,
        video_dir=video_path.parent,
    )
    shots = shot_extractor.extract(video_path.stem)

    keyframe_extractor = KeyframeExtractor(
        video_dir=video_path.parent,
        keyframe_dir=output_dir / "keyframes",
    )
    frames = keyframe_extractor.extract_for_video(
        video_path.stem,
        shots,
        existing_frame_idxs=_parse_existing_frame_indices(args.existing_frame_indices),
    )

    shots_path = output_dir / f"{video_path.stem}_shots.json"
    keyframes_path = output_dir / f"{video_path.stem}_additional_keyframes.json"
    shots_path.write_text(
        json.dumps([_shot_to_dict(shot) for shot in shots], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    keyframes_path.write_text(
        json.dumps([_frame_to_dict(frame) for frame in frames], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Detected {len(shots)} shots and extracted {len(frames)} additional keyframes.")
    print(f"Saved shot metadata to: {shots_path}")
    print(f"Saved keyframe metadata to: {keyframes_path}")
    print(f"Saved JPEG images to: {output_dir / 'keyframes' / video_path.stem}")


if __name__ == "__main__":
    main()
