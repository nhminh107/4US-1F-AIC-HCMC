"""Run one Open Images detection batch from an image folder.

Example:
    python -m BackEnd.tests.object_detection.smoke_detection \
        --folder data/keyframes/L21_V014 \
        --batch-size 8 \
        --device /GPU:0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.openimages_jsonl import detect_frames
from BackEnd.app.object_detection.preprocess import load_image
from BackEnd.app.object_detection.tfhub_openimages_detector import (
    MODEL_URL,
    TFHubOpenImagesDetector,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_batch(folder: Path, batch_size: int) -> list[Path]:
    """Return the first naturally sorted image batch in a folder."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {folder}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    image_paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=_natural_image_key)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {folder}")
    return image_paths[:batch_size]


def build_test_frames(image_paths: list[Path], folder: Path) -> list[FrameMetadata]:
    """Create the minimum pipeline contracts required by object detection.

    Object detection only consumes ``frame_id`` and ``frame_path``. The temporal
    fields below are deterministic placeholders for this isolated smoke test and
    are not written to the application database.
    """
    video_id = folder.name
    frames: list[FrameMetadata] = []
    for position, image_path in enumerate(image_paths, start=1):
        frame_number = int(image_path.stem) if image_path.stem.isdigit() else position
        frames.append(
            FrameMetadata(
                frame_id=f"{video_id}_{frame_number:03d}",
                video_id=video_id,
                shot_id=f"{video_id}_TEST",
                timestamp_ms=0,
                fps=0.0,
                frame_idx=frame_number,
                frame_role="keyframe",
                source="official",
                n=frame_number,
                frame_path=image_path.resolve(),
            )
        )
    return frames


def validate_batch_shape(image_paths: list[Path]) -> tuple[int, int, int]:
    """Require one image shape so the pipeline creates one detector batch."""
    shapes = {load_image(str(image_path)).shape for image_path in image_paths}
    if len(shapes) != 1:
        formatted_shapes = ", ".join(str(shape) for shape in sorted(shapes))
        raise ValueError(
            "The selected images have different shapes, so they cannot form one "
            f"detector batch. Found: {formatted_shapes}"
        )
    return next(iter(shapes))


def print_results(
    frames: list[FrameMetadata],
    results: list[ObjectDetectionResult],
) -> None:
    """Print detection contracts grouped by input image."""
    results_by_frame: dict[str, list[ObjectDetectionResult]] = defaultdict(list)
    for result in results:
        results_by_frame[result.frame_id].append(result)

    print("\nDetection results")
    print("=" * 80)
    for frame in frames:
        frame_results = sorted(
            results_by_frame[frame.frame_id],
            key=lambda item: item.confidence,
            reverse=True,
        )
        print(
            f"{frame.frame_path.name} | frame_id={frame.frame_id} | "
            f"objects={len(frame_results)}"
        )
        for result in frame_results[:10]:
            bbox = (
                f"({result.x_min:.3f}, {result.y_min:.3f}) - "
                f"({result.x_max:.3f}, {result.y_max:.3f})"
            )
            print(
                f"  MID={result.class_id:<12} "
                f"confidence={result.confidence:.3f} bbox_norm={bbox}"
            )
        if len(frame_results) > 10:
            print(f"  ... and {len(frame_results) - 10} more object(s)")


def save_results(
    output_path: Path,
    frames: list[FrameMetadata],
    results: list[ObjectDetectionResult],
) -> None:
    """Save input frame information and output contracts as readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_frames": [
            {
                "frame_id": frame.frame_id,
                "frame_path": str(frame.frame_path),
            }
            for frame in frames
        ],
        "detections": [asdict(result) for result in results],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _natural_image_key(path: Path) -> tuple[int, int | str, str]:
    """Sort numeric file names numerically and other names alphabetically."""
    if path.stem.isdigit():
        return (0, int(path.stem), path.name.lower())
    return (1, path.stem.lower(), path.name.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one logical Open Images batch from a folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing the images for this test batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum number of images selected from the folder (default: 8).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional TensorFlow device, for example /GPU:0 or /CPU:0.",
    )
    parser.add_argument(
        "--model-source",
        default=MODEL_URL,
        help="TF Hub URL or local SavedModel path.",
    )
    parser.add_argument(
        "--model-sha256",
        default=None,
        help="Expected SHA-256 when --model-source is a local path.",
    )
    parser.add_argument(
        "--require-local-model",
        action="store_true",
        help="Reject remote model URLs and require a local model path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Without this option, only print results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_batch(args.folder, args.batch_size)
    batch_shape = validate_batch_shape(image_paths)
    frames = build_test_frames(image_paths, args.folder)

    print(f"Folder: {args.folder.resolve()}")
    print(f"Requested batch size: {args.batch_size}")
    print(f"Selected images: {len(frames)}")
    print(f"Shared image shape: {batch_shape}")
    for frame in frames:
        print(f"  - {frame.frame_path.name}")

    detector = TFHubOpenImagesDetector(
        model_url=args.model_source,
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
        device=args.device,
        expected_model_sha256=args.model_sha256,
        require_local_model=args.require_local_model,
    )
    execution_mode = (
        "native model batch"
        if detector.supports_batching
        else "logical batch with sequential model fallback"
    )
    print(f"Model execution mode: {execution_mode}")

    started_at = time.perf_counter()
    results = detect_frames(
        frames,
        detector=detector,
        batch_size=args.batch_size,
    )
    elapsed_seconds = time.perf_counter() - started_at

    print_results(frames, results)
    print("=" * 80)
    print(
        f"Completed: {len(frames)} image(s), {len(results)} object(s), "
        f"{elapsed_seconds:.2f} second(s)"
    )

    if args.output is not None:
        save_results(args.output, frames, results)
        print(f"Saved JSON: {args.output.resolve()}")
    else:
        print("Saved JSON: no (--output was not provided)")


if __name__ == "__main__":
    main()
