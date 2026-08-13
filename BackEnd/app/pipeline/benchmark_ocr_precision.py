"""Benchmark OCR precision without modifying PostgreSQL records."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from BackEnd.CONFIG import OCRConfig
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.ocr.service import OCRService


def run_benchmark(
    *,
    video_id: str,
    precision: str,
    output_path: Path,
    frame_limit: int | None,
    frame_chunk_size: int,
    detection_batch_size: int,
    recognition_batch_size: int,
) -> dict[str, object]:
    """Run read-only OCR inference and persist comparable benchmark output."""

    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive when provided.")
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OCR precision benchmark.")

    database = PostgreManager()
    service = OCRService(
        config=OCRConfig(
            device="gpu:0",
            precision=precision,
            detection_batch_size=detection_batch_size,
            recognition_batch_size=recognition_batch_size,
        )
    )
    try:
        frames = [
            frame
            for frame in database.get_frame_record_by_video_id(video_id)
            if frame.source == "official"
        ]
        if frame_limit is not None:
            frames = frames[:frame_limit]
        if not frames:
            raise ValueError(f"No official frames found for video '{video_id}'.")

        torch.cuda.reset_peak_memory_stats()
        started_at = time.perf_counter()
        results = []
        for start in range(0, len(frames), frame_chunk_size):
            results.extend(service.process_batch(frames[start : start + frame_chunk_size]))
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started_at

        texts_by_frame: dict[str, list[str]] = {frame.frame_id: [] for frame in frames}
        for result in results:
            texts_by_frame[result.frame_id].append(result.text)

        payload: dict[str, object] = {
            "video_id": video_id,
            "precision": precision,
            "frame_count": len(frames),
            "row_count": len(results),
            "elapsed_seconds": elapsed_seconds,
            "frames_per_second": len(frames) / elapsed_seconds,
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "texts_by_frame": texts_by_frame,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload
    finally:
        service.close()
        database.engine.dispose()


def compare_results(fp32_path: Path, fp16_path: Path) -> dict[str, object]:
    """Compare frame-level OCR text produced by two benchmark runs."""

    fp32 = json.loads(fp32_path.read_text(encoding="utf-8"))
    fp16 = json.loads(fp16_path.read_text(encoding="utf-8"))
    if fp32["video_id"] != fp16["video_id"]:
        raise ValueError("Benchmark video IDs differ.")
    if fp32["frame_count"] != fp16["frame_count"]:
        raise ValueError("Benchmark frame counts differ.")

    fp32_texts = fp32["texts_by_frame"]
    fp16_texts = fp16["texts_by_frame"]
    frame_ids = list(fp32_texts)
    exact_matches = sum(
        fp32_texts[frame_id] == fp16_texts.get(frame_id) for frame_id in frame_ids
    )
    fp32_counter = Counter(
        text for texts in fp32_texts.values() for text in texts
    )
    fp16_counter = Counter(
        text for texts in fp16_texts.values() for text in texts
    )
    shared_rows = sum((fp32_counter & fp16_counter).values())
    union_rows = sum((fp32_counter | fp16_counter).values())
    return {
        "video_id": fp32["video_id"],
        "frame_count": len(frame_ids),
        "fp32_seconds": fp32["elapsed_seconds"],
        "fp16_seconds": fp16["elapsed_seconds"],
        "speedup": fp32["elapsed_seconds"] / fp16["elapsed_seconds"],
        "fp32_rows": fp32["row_count"],
        "fp16_rows": fp16["row_count"],
        "exact_frame_ratio": exact_matches / len(frame_ids),
        "text_multiset_jaccard": shared_rows / union_rows if union_rows else 1.0,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--video-id", required=True)
    run_parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--frame-limit", type=int, default=64)
    run_parser.add_argument("--frame-chunk-size", type=int, default=32)
    run_parser.add_argument("--detection-batch-size", type=int, default=24)
    run_parser.add_argument("--recognition-batch-size", type=int, default=192)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--fp32", type=Path, required=True)
    compare_parser.add_argument("--fp16", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "compare":
        report = compare_results(arguments.fp32, arguments.fp16)
    else:
        report = run_benchmark(
            video_id=arguments.video_id,
            precision=arguments.precision,
            output_path=arguments.output,
            frame_limit=arguments.frame_limit,
            frame_chunk_size=arguments.frame_chunk_size,
            detection_batch_size=arguments.detection_batch_size,
            recognition_batch_size=arguments.recognition_batch_size,
        )
        report = {key: value for key, value in report.items() if key != "texts_by_frame"}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
