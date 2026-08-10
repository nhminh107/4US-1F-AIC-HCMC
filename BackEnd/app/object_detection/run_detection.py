"""Run object detection over keyframes."""

from __future__ import annotations

import argparse
import os
from typing import Optional

from BackEnd.CONFIG import (
    OBJECT_DETECTION_CHUNK_SIZE as CHUNK_SIZE,
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    OBJECT_DETECTION_KEYFRAMES_DIR,
    OBJECT_DETECTION_NMS_IOU_THRESHOLD,
    OBJECT_DETECTION_OUTPUT_PATH,
)
from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.exporter import export_results_json
from BackEnd.app.object_detection.postprocess import clip_detections, nms
from BackEnd.app.object_detection.preprocess import preprocess
from BackEnd.app.object_detection.schemas import FrameDetectionResult
from BackEnd.app.object_detection.utils import default_frame_id, iter_chunks, scan_keyframes
from BackEnd.app.object_detection.yolo_detector import YOLODetector

KEYFRAMES_DIR = str(OBJECT_DETECTION_KEYFRAMES_DIR)
OUTPUT_PATH = str(OBJECT_DETECTION_OUTPUT_PATH)


def run_detection_on_chunk(
    img_paths: list[str],
    detector: Detector,
    *,
    keyframes_dir: str = KEYFRAMES_DIR,
    max_side: int | None = None,
    apply_nms: bool = False,
    nms_iou_threshold: float = OBJECT_DETECTION_NMS_IOU_THRESHOLD,
) -> list[FrameDetectionResult]:
    results: list[FrameDetectionResult] = []

    for img_path in img_paths:
        image_path = os.path.join(keyframes_dir, img_path)
        image = preprocess(image_path, max_side=max_side)
        image_height, image_width = image.shape[:2]
        frame_id = default_frame_id(img_path)

        detections = detector.detect(image, frame_id=frame_id, img_path=img_path)
        detections = clip_detections(
            detections,
            image_width=image_width,
            image_height=image_height,
        )
        if apply_nms:
            detections = nms(detections, iou_threshold=nms_iou_threshold)

        results.append(
            FrameDetectionResult(
                img_path=img_path,
                frame_id=frame_id,
                image_width=image_width,
                image_height=image_height,
                detections=detections,
            )
        )

    return results


def run_detection(
    img_paths: Optional[list[str]] = None,
    *,
    keyframes_dir: str = KEYFRAMES_DIR,
    output_path: str = OUTPUT_PATH,
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    iou_threshold: float = OBJECT_DETECTION_NMS_IOU_THRESHOLD,
    device: str | None = None,
    class_names: list[str] | None = None,
    class_ids: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_side: int | None = None,
) -> list[FrameDetectionResult]:
    img_paths = img_paths if img_paths is not None else scan_keyframes(keyframes_dir)
    detector = YOLODetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
        class_names=class_names,
        class_ids=class_ids,
    )

    all_results: list[FrameDetectionResult] = []
    chunks = list(iter_chunks(img_paths, chunk_size))
    for chunk_idx, chunk in enumerate(chunks, start=1):
        chunk_results = run_detection_on_chunk(
            chunk,
            detector,
            keyframes_dir=keyframes_dir,
            max_side=max_side,
        )
        all_results.extend(chunk_results)
        export_results_json(all_results, output_path)
        print(
            f"[object_detection] chunk {chunk_idx}/{len(chunks)} "
            f"({len(chunk)} keyframe) -> total={len(all_results)}"
        )

    total_objects = sum(len(item.detections) for item in all_results)
    print(
        f"[object_detection] done: {len(all_results)} images, "
        f"{total_objects} objects -> {output_path}"
    )
    return all_results


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO object detection.")
    parser.add_argument("--keyframes-dir", default=KEYFRAMES_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--confidence", type=float, default=OBJECT_DETECTION_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--iou", type=float, default=OBJECT_DETECTION_NMS_IOU_THRESHOLD)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", default=None, help="Comma-separated class names.")
    parser.add_argument("--class-ids", default=None, help="Comma-separated ids, e.g. c000,c002.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    img_paths = scan_keyframes(args.keyframes_dir)
    if args.limit:
        img_paths = img_paths[: args.limit]

    run_detection(
        img_paths=img_paths,
        keyframes_dir=args.keyframes_dir,
        output_path=args.output,
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        class_names=_parse_csv(args.classes),
        class_ids=_parse_csv(args.class_ids),
        chunk_size=args.chunk_size,
        max_side=args.max_side,
    )


if __name__ == "__main__":
    main()
