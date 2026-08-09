"""CLI for one-image Faster R-CNN Open Images inference to contract JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from BackEnd.app.object_detection.openimages_jsonl import detect_image_to_jsonl
from BackEnd.app.object_detection.tfhub_openimages_detector import TFHubOpenImagesDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN Inception-ResNet-v2 Open Images inference for one image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    detector = TFHubOpenImagesDetector(confidence_threshold=args.confidence)
    contracts = detect_image_to_jsonl(
        args.image,
        args.output,
        detector=detector,
        frame_id=args.frame_id,
    )
    print(f"Wrote {len(contracts)} ObjectDetectionResult records to {args.output}.")


if __name__ == "__main__":
    main()
