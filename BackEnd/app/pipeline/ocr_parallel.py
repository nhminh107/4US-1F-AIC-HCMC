"""Run OCR over non-overlapping video shards in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Sequence

from sqlalchemy import delete, select

from BackEnd.app.database.models import Frame, OCR
from BackEnd.app.database.postgre_db import PostgreManager

COMMITTED_VIDEO_PATTERN = re.compile(r"\[ocr-worker\].* committed (\S+),")


def _split_round_robin(video_ids: list[str], worker_count: int) -> list[list[str]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    return [video_ids[index::worker_count] for index in range(worker_count)]


def _load_completed_video_ids(resume_logs: Sequence[Path]) -> set[str]:
    """Load committed video IDs from every available append-only worker log."""

    completed: set[str] = set()
    for resume_log in resume_logs:
        if resume_log.is_file():
            completed.update(
                COMMITTED_VIDEO_PATTERN.findall(
                    resume_log.read_text(encoding="utf-8", errors="replace")
                )
            )
    return completed


def run_parallel_ocr(
    *,
    worker_count: int,
    result_dir: Path,
    frame_source: str,
    precision: str = "fp32",
    frame_chunk_size: int,
    detection_batch_size: int,
    recognition_batch_size: int,
    resume_log: Path | None = None,
    resume_logs: Sequence[Path] = (),
) -> None:
    """Launch OCR workers and require every shard to finish successfully."""

    database = PostgreManager()
    try:
        video_ids = [video.video_id for video in database.get_list_video()]
        selected_resume_logs = list(resume_logs)
        if resume_log is not None:
            selected_resume_logs.append(resume_log)
        completed_video_ids = _load_completed_video_ids(selected_resume_logs)
        video_ids = [
            video_id for video_id in video_ids if video_id not in completed_video_ids
        ]
        if video_ids:
            with database.session_factory.begin() as session:
                pending_frame_ids = select(Frame.frame_id).where(
                    Frame.video_id.in_(video_ids)
                )
                session.execute(delete(OCR).where(OCR.frame_id.in_(pending_frame_ids)))
    finally:
        database.engine.dispose()

    print(
        f"[ocr-parallel] resume completed={len(completed_video_ids)}, "
        f"pending={len(video_ids)}",
        flush=True,
    )
    if not video_ids:
        print("[ocr-parallel] no pending videos", flush=True)
        return

    shards = [shard for shard in _split_round_robin(video_ids, worker_count) if shard]
    result_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    for worker_index, shard in enumerate(shards, start=1):
        result_path = result_dir / f"worker-{worker_index}.json"
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "BackEnd.app.pipeline.ocr_worker",
            "--result-path",
            str(result_path),
            "--frame-source",
            frame_source,
            "--precision",
            precision,
            "--frame-chunk-size",
            str(frame_chunk_size),
            "--detection-batch-size",
            str(detection_batch_size),
            "--recognition-batch-size",
            str(recognition_batch_size),
            *shard,
        ]
        print(
            f"[ocr-parallel] worker={worker_index}, videos={len(shard)}, "
            f"first={shard[0]}, last={shard[-1]}",
            flush=True,
        )
        processes.append((subprocess.Popen(command), result_path, len(shard)))

    failures: list[str] = []
    for process, result_path, expected_count in processes:
        return_code = process.wait()
        if not result_path.is_file():
            failures.append(f"pid={process.pid}: missing result, returncode={return_code}")
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        completed = payload.get("completed_video_ids", [])
        if return_code != 0 or payload.get("error") or len(completed) != expected_count:
            failures.append(
                f"pid={process.pid}: returncode={return_code}, "
                f"completed={len(completed)}/{expected_count}, error={payload.get('error')}"
            )
    if failures:
        raise RuntimeError("OCR worker failure(s): " + "; ".join(failures))
    print("[ocr-parallel] all workers completed", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--result-dir", type=Path, default=Path("artifacts/ocr-results"))
    parser.add_argument("--frame-source", default="official")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--frame-chunk-size", type=int, default=64)
    parser.add_argument("--detection-batch-size", type=int, default=32)
    parser.add_argument("--recognition-batch-size", type=int, default=256)
    parser.add_argument(
        "--resume-log",
        type=Path,
        action="append",
        default=[],
        help="Repeat to combine committed video IDs from multiple prior runs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_parallel_ocr(
        worker_count=arguments.workers,
        result_dir=arguments.result_dir,
        frame_source=arguments.frame_source,
        precision=arguments.precision,
        frame_chunk_size=arguments.frame_chunk_size,
        detection_batch_size=arguments.detection_batch_size,
        recognition_batch_size=arguments.recognition_batch_size,
        resume_logs=arguments.resume_log,
    )


if __name__ == "__main__":
    main()
