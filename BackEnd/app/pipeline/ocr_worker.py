"""One-shot OCR subprocess that avoids Paddle native-runtime teardown faults."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

from BackEnd.CONFIG import OCRConfig


def _ocr_config_for_retry(
    retry_level: int,
    *,
    precision: str = "fp32",
    detection_batch_size: int | None = None,
    recognition_batch_size: int | None = None,
) -> OCRConfig:
    if retry_level < 0:
        raise ValueError("retry_level must not be negative.")
    base = OCRConfig(device="gpu:0", precision=precision)
    divisor = 2**retry_level
    detection_batch_size = detection_batch_size or base.detection_batch_size
    recognition_batch_size = recognition_batch_size or base.recognition_batch_size
    return replace(
        base,
        detection_batch_size=max(1, detection_batch_size // divisor),
        recognition_batch_size=max(1, recognition_batch_size // divisor),
    )


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run(
    video_ids: list[str],
    retry_level: int,
    *,
    frame_source: str | None = None,
    precision: str = "fp32",
    frame_chunk_size: int,
    detection_batch_size: int,
    recognition_batch_size: int,
) -> dict[str, Any]:
    """Run OCR, committing each video before returning a serializable status."""

    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.ocr.service import OCRService
    from BackEnd.app.pipeline.ocr import run_ocr

    db = PostgreManager()
    service = OCRService(
        config=_ocr_config_for_retry(
            retry_level,
            precision=precision,
            detection_batch_size=detection_batch_size,
            recognition_batch_size=recognition_batch_size,
        )
    )
    completed: list[str] = []
    try:
        total = len(video_ids)
        for position, video_id in enumerate(video_ids, start=1):
            try:
                results = run_ocr(
                    video_id,
                    db,
                    service,
                    frame_chunk_size=frame_chunk_size,
                    frame_source=frame_source,
                )
            except Exception as error:
                message = str(error)
                return {
                    "completed_video_ids": completed,
                    "oom_video_id": video_id if _is_gpu_oom(message) else None,
                    "error": message,
                }
            completed.append(video_id)
            print(
                f"[ocr-worker] {position}/{total} committed {video_id}, "
                f"rows={len(results)}, source={frame_source or 'all'}",
                flush=True,
            )
        return {"completed_video_ids": completed, "error": None}
    finally:
        close = getattr(service, "close", None)
        if callable(close):
            close()
        db.engine.dispose()


def _is_gpu_oom(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda error: memory",
            "cuda out of memory",
            "resourceexhausted",
            "resource exhausted",
        )
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR in an isolated process.")
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--retry-level", type=int, default=0)
    parser.add_argument("--frame-source", default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--frame-chunk-size", type=int, default=32)
    parser.add_argument("--detection-batch-size", type=int, default=16)
    parser.add_argument("--recognition-batch-size", type=int, default=128)
    parser.add_argument("video_ids", nargs="+")
    return parser.parse_args()


def main() -> None:
    """Persist the worker result, then bypass Paddle's faulty interpreter teardown."""

    arguments = parse_arguments()
    try:
        payload = run(
            arguments.video_ids,
            arguments.retry_level,
            frame_source=arguments.frame_source,
            precision=arguments.precision,
            frame_chunk_size=arguments.frame_chunk_size,
            detection_batch_size=arguments.detection_batch_size,
            recognition_batch_size=arguments.recognition_batch_size,
        )
    except BaseException as error:
        payload = {"completed_video_ids": [], "oom_video_id": None, "error": str(error)}

    _write_result(arguments.result_path, payload)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if payload["error"] is None else 1)


if __name__ == "__main__":
    main()
