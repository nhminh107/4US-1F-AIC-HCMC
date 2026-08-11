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


def _ocr_config_for_retry(retry_level: int) -> OCRConfig:
    if retry_level < 0:
        raise ValueError("retry_level must not be negative.")
    base = OCRConfig(device="gpu:0")
    divisor = 2**retry_level
    return replace(
        base,
        detection_batch_size=max(1, base.detection_batch_size // divisor),
        recognition_batch_size=max(1, base.recognition_batch_size // divisor),
    )


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run(video_ids: list[str], retry_level: int) -> dict[str, Any]:
    """Run OCR, committing each video before returning a serializable status."""

    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.ocr.service import OCRService
    from BackEnd.app.pipeline.ocr import run_ocr

    db = PostgreManager()
    service = OCRService(config=_ocr_config_for_retry(retry_level))
    completed: list[str] = []
    try:
        for video_id in video_ids:
            try:
                run_ocr(video_id, db, service)
            except Exception as error:
                message = str(error)
                return {
                    "completed_video_ids": completed,
                    "oom_video_id": video_id if _is_gpu_oom(message) else None,
                    "error": message,
                }
            completed.append(video_id)
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
    parser.add_argument("video_ids", nargs="+")
    return parser.parse_args()


def main() -> None:
    """Persist the worker result, then bypass Paddle's faulty interpreter teardown."""

    arguments = parse_arguments()
    try:
        payload = run(arguments.video_ids, arguments.retry_level)
    except BaseException as error:
        payload = {"completed_video_ids": [], "oom_video_id": None, "error": str(error)}

    _write_result(arguments.result_path, payload)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if payload["error"] is None else 1)


if __name__ == "__main__":
    main()
