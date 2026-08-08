"""Quick object-detection smoke test on a subset of keyframes.

Usage:
    python -m BackEnd.app.object_detection.test_detection --limit 20
    python -m BackEnd.app.object_detection.test_detection --video K01_V001 --limit 20
    python -m BackEnd.app.object_detection.test_detection --classes person,car
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Optional

from BackEnd.app.object_detection.run_detection import KEYFRAMES_DIR, run_detection_on_chunk
from BackEnd.app.object_detection.utils import scan_keyframes
from BackEnd.app.object_detection.yolo_detector import YOLODetector

_SHOT_NUM_RE = re.compile(r"S(\d+)")


def _shot_number(img_path: str) -> Optional[int]:
    parts = img_path.split("/")
    if len(parts) < 2:
        return None
    match = _SHOT_NUM_RE.fullmatch(parts[1])
    return int(match.group(1)) if match else None


def filter_keyframes(
    img_paths: list[str],
    *,
    video: str | None = None,
    shot_min: int | None = None,
    shot_max: int | None = None,
    limit: int | None = None,
) -> list[str]:
    filtered = img_paths
    if video:
        filtered = [path for path in filtered if path.startswith(f"{video}/")]
    if shot_min is not None or shot_max is not None:
        kept: list[str] = []
        for img_path in filtered:
            shot_num = _shot_number(img_path)
            if shot_num is None:
                continue
            if shot_min is not None and shot_num < shot_min:
                continue
            if shot_max is not None and shot_num > shot_max:
                continue
            kept.append(img_path)
        filtered = kept
    if limit:
        filtered = filtered[:limit]
    return filtered


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test object detection on keyframes.")
    parser.add_argument("--keyframes-dir", default=KEYFRAMES_DIR)
    parser.add_argument("--video", default=None)
    parser.add_argument("--shot-min", type=int, default=None)
    parser.add_argument("--shot-max", type=int, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()

    img_paths = filter_keyframes(
        scan_keyframes(args.keyframes_dir),
        video=args.video,
        shot_min=args.shot_min,
        shot_max=args.shot_max,
        limit=args.limit,
    )
    if not img_paths:
        print("No keyframes matched the selected filters.")
        return

    print(f"[test_object_detection] running on {len(img_paths)} frame(s)...")
    start = time.time()
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        class_names=_parse_csv(args.classes),
    )
    results = run_detection_on_chunk(
        img_paths,
        detector,
        keyframes_dir=args.keyframes_dir,
    )

    elapsed = time.time() - start
    total_objects = sum(len(result.detections) for result in results)
    print(
        f"[test_object_detection] done after {elapsed:.1f}s: "
        f"{len(results)} images, {total_objects} objects."
    )
    for result in results[:20]:
        names = ", ".join(
            f"{detection.class_name}:{detection.confidence:.2f}"
            for detection in result.detections[:8]
        )
        print(f"  {result.img_path}: {len(result.detections)} object(s) {names}")


if __name__ == "__main__":
    main()
