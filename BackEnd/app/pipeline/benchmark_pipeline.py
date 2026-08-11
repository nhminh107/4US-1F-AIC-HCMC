"""Run the real write-enabled offline pipeline and record a GPU benchmark.

This is intended for a disposable/staging database. It performs the same
PostgreSQL and FAISS writes as the production pipeline and never deletes or
resets existing records.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
import time
from typing import Callable

from BackEnd import CONFIG
from BackEnd.app.contracts.pipeline import VideoMetadata
from BackEnd.app.database.faiss_db import FAISS_Manager
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.pipeline.main_pipeline import Pipeline
from BackEnd.app.pipeline.parallel_runner import check_parallel_gpu_preflight


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    name: str
    elapsed_seconds: float


class GPUMonitor:
    """Poll NVIDIA-SMI without adding a runtime dependency."""

    def __init__(self, output_path: Path, interval_seconds: float = 1.0) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 2.0)

    def _run(self) -> None:
        with self.output_path.open("a", encoding="utf-8") as output:
            while not self._stop_event.is_set():
                sample = _read_gpu_sample()
                if sample is not None:
                    output.write(json.dumps(sample) + "\n")
                    output.flush()
                self._stop_event.wait(self.interval_seconds)


def _read_gpu_sample() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    fields = (
        "timestamp",
        "gpu_index",
        "gpu_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_watts",
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if not rows:
        return None
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        return None
    return dict(zip(fields, values, strict=True))


def _run_stage(
    name: str,
    callback: Callable[[], None],
    measurements: list[StageMeasurement],
) -> None:
    started_at = time.perf_counter()
    callback()
    measurements.append(
        StageMeasurement(name=name, elapsed_seconds=time.perf_counter() - started_at)
    )


def _select_videos(
    db: PostgreManager,
    video_ids: list[str],
) -> list[VideoMetadata]:
    available = {video.video_id: video for video in db.get_list_video()}
    missing = [video_id for video_id in video_ids if video_id not in available]
    if missing:
        raise ValueError(f"Videos do not exist in PostgreSQL: {missing}")
    return [available[video_id] for video_id in video_ids]


def _summarize_gpu_samples(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {"sample_count": 0}
    samples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        return {"sample_count": 0}

    def numeric_values(field: str) -> list[float]:
        values: list[float] = []
        for sample in samples:
            try:
                values.append(float(sample[field]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    memory = numeric_values("memory_used_mib")
    utilization = numeric_values("gpu_utilization_percent")
    return {
        "sample_count": len(samples),
        "max_memory_used_mib": max(memory, default=0.0),
        "average_gpu_utilization_percent": (
            sum(utilization) / len(utilization) if utilization else 0.0
        ),
    }


def run_benchmark(arguments: argparse.Namespace) -> Path:
    """Execute selected production stages and persist a benchmark report."""

    if arguments.duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if not arguments.video_id:
        raise ValueError("At least one --video-id is required.")

    preflight = check_parallel_gpu_preflight()
    if arguments.parallel_mode == "parallel" and not preflight.parallel_safe:
        raise RuntimeError(f"Parallel GPU benchmark cannot start: {preflight.reason}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(arguments.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    gpu_samples_path = output_dir / "gpu_samples.jsonl"
    report_path = output_dir / "report.json"

    db = PostgreManager()
    try:
        videos = _select_videos(db, arguments.video_id)
        faiss = FAISS_Manager(
            img_dim=CONFIG.CLIP_DIMENSION,
            clip_dim=CONFIG.CLIP_DIMENSION,
            shot_dim=CONFIG.CLIP_DIMENSION,
            model_name=CONFIG.CLIP_MODEL,
            data_path=Path(arguments.faiss_dir) if arguments.faiss_dir else output_dir / "faiss",
            load_existing=arguments.load_existing_faiss,
        )
        pipeline = Pipeline(
            db=db,
            faiss=faiss,
            videos=videos,
            parallel_mode=arguments.parallel_mode,
        )
        measurements: list[StageMeasurement] = []
        monitor = GPUMonitor(gpu_samples_path, arguments.sample_interval_seconds)
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started_at = time.perf_counter()
        monitor.start()
        error: BaseException | None = None
        try:
            _run_stage("shot_extraction", pipeline.run_shot_extraction, measurements)
            _run_stage("parallel_processing", pipeline.run_dependent_stages, measurements)
            if time.perf_counter() - started_at < arguments.duration_seconds:
                _run_stage("embeddings", pipeline.run_embeddings, measurements)
        except BaseException as caught_error:
            error = caught_error
        finally:
            monitor.stop()

        elapsed_seconds = time.perf_counter() - started_at
        report = {
            "run_id": run_id,
            "started_at_utc": started_at_utc,
            "requested_video_ids": arguments.video_id,
            "parallel_mode": arguments.parallel_mode,
            "duration_limit_seconds": arguments.duration_seconds,
            "elapsed_seconds": elapsed_seconds,
            "stages": [asdict(measurement) for measurement in measurements],
            "preflight": asdict(preflight),
            "faiss_directory": str(faiss.datapath),
            "gpu": _summarize_gpu_samples(gpu_samples_path),
            "error": str(error) if error is not None else None,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if error is not None:
            raise error
        return report_path
    finally:
        db.engine.dispose()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the write-enabled pipeline benchmark on a staging database."
    )
    parser.add_argument(
        "--video-id",
        action="append",
        required=True,
        help="Video ID already present in PostgreSQL. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=3_600,
        help="Stop before starting a new stage once this budget is reached.",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("parallel", "auto", "sequential"),
        default="parallel",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
        help="Optional existing FAISS directory. Omit for an isolated test index.",
    )
    parser.add_argument("--load-existing-faiss", action="store_true")
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    report = run_benchmark(parse_arguments())
    print(f"Benchmark report: {report}")
