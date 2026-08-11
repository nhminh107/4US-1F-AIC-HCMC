"""Isolated GPU workers for the offline preprocessing pipeline.

Workers intentionally create their own database engine and models after the
``spawn`` boundary. CUDA runtimes and SQLAlchemy connections must not be
inherited from the parent process.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
import os
from typing import Literal

from BackEnd.CONFIG import (
    PIPELINE_GPU_HEADROOM_GIB,
    PIPELINE_OCR_VRAM_BUDGET_GIB,
    PIPELINE_TRACKING_VRAM_BUDGET_GIB,
    OCRConfig,
)

WorkerKind = Literal["tracking", "enrichment"]


class GPUPreflightError(RuntimeError):
    """Raised when explicitly requested parallel GPU execution is unavailable."""


@dataclass(frozen=True, slots=True)
class GPUPreflight:
    parallel_safe: bool
    reason: str


@dataclass(frozen=True, slots=True)
class WorkerResult:
    kind: WorkerKind
    completed_video_ids: tuple[str, ...]
    oom_video_id: str | None = None
    oom_message: str | None = None

    # This keeps retry scheduling local to the orchestrator without
    # serializing database or model objects.
    requested_video_ids: tuple[str, ...] = ()


def check_parallel_gpu_preflight() -> GPUPreflight:
    """Return whether tracking and hybrid OCR can safely share one GPU."""

    try:
        import torch
    except ImportError:
        return GPUPreflight(False, "PyTorch is not installed.")
    if not torch.cuda.is_available():
        return GPUPreflight(False, "PyTorch cannot access CUDA.")

    try:
        import paddle
    except ImportError:
        return GPUPreflight(False, "PaddlePaddle is not installed.")
    if not (
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
        return GPUPreflight(False, "PaddlePaddle cannot access CUDA.")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    required_gib = (
        PIPELINE_TRACKING_VRAM_BUDGET_GIB
        + PIPELINE_OCR_VRAM_BUDGET_GIB
        + PIPELINE_GPU_HEADROOM_GIB
    )
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    if free_gib < required_gib:
        return GPUPreflight(
            False,
            f"Only {free_gib:.1f} GiB VRAM is free; parallel workers require "
            f"at least {required_gib} GiB (total={total_gib:.1f} GiB).",
        )
    return GPUPreflight(True, f"{free_gib:.1f}/{total_gib:.1f} GiB VRAM is free.")


def run_parallel_workers(
    tracking_video_ids: tuple[str, ...],
    enrichment_video_ids: tuple[str, ...],
) -> tuple[WorkerResult, WorkerResult]:
    """Run exactly one tracking and one enrichment worker in fresh processes."""

    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        tracking_future = executor.submit(
            run_tracking_worker,
            tracking_video_ids,
        )
        enrichment_future = executor.submit(
            run_enrichment_worker,
            enrichment_video_ids,
            0,
        )
        return tracking_future.result(), enrichment_future.result()


def run_worker_exclusively(
    kind: WorkerKind,
    video_ids: tuple[str, ...],
    *,
    ocr_retry_level: int = 0,
) -> WorkerResult:
    """Run a retry in a new, single worker process to reclaim CUDA memory."""

    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        if kind == "tracking":
            future = executor.submit(run_tracking_worker, video_ids)
        else:
            future = executor.submit(
                run_ocr_worker,
                video_ids,
                ocr_retry_level,
            )
        return future.result()


def run_tracking_worker(video_ids: tuple[str, ...]) -> WorkerResult:
    """Persist tracking results for the assigned videos in one CUDA process."""

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.pipeline.tracking import track_video
    from BackEnd.app.tracking.tracking import YOLOTrackingService

    db = PostgreManager()
    tracker = YOLOTrackingService()
    completed: list[str] = []
    try:
        videos = {video.video_id: video for video in db.get_list_video()}
        for video_id in video_ids:
            try:
                track_video(videos[video_id], db, tracker)
            except Exception as error:
                if _is_gpu_oom(error):
                    return WorkerResult(
                        kind="tracking",
                        completed_video_ids=tuple(completed),
                        oom_video_id=video_id,
                        oom_message=str(error),
                        requested_video_ids=video_ids,
                    )
                raise
            completed.append(video_id)
    finally:
        _close_resource(tracker)
        db.engine.dispose()
    return WorkerResult(
        kind="tracking",
        completed_video_ids=tuple(completed),
        requested_video_ids=video_ids,
    )


def run_enrichment_worker(
    video_ids: tuple[str, ...],
    ocr_retry_level: int,
) -> WorkerResult:
    """Run keyframe, clip and OCR stages per video in one CUDA process."""

    # Must be set before Paddle is imported by OCRService.
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from BackEnd.app.clip_extractor import ClipExtractor
    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.keyframe_extractor import KeyframeExtractor
    from BackEnd.app.pipeline.extract_clip import extract_clip
    from BackEnd.app.pipeline.extract_keyframe import extract_keyframes

    db = PostgreManager()
    keyframe_extractor = KeyframeExtractor()
    clip_extractor = ClipExtractor()
    try:
        for video_id in video_ids:
            extract_keyframes(video_id, db, keyframe_extractor)
            extract_clip(video_id, db, clip_extractor)
    finally:
        _close_resource(clip_extractor)
        _close_resource(keyframe_extractor)
        db.engine.dispose()

    # OCR is deliberately separate from persistence prep so an OOM retry does
    # not recreate already persisted frames or clip rows.
    return run_ocr_worker(video_ids, ocr_retry_level)


def run_ocr_worker(
    video_ids: tuple[str, ...],
    ocr_retry_level: int,
) -> WorkerResult:
    """Run only OCR, allowing an OOM retry without duplicate frame/clip rows."""

    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.ocr.service import OCRService
    from BackEnd.app.pipeline.ocr import run_ocr

    db = PostgreManager()
    ocr_service = OCRService(config=_ocr_config_for_retry(ocr_retry_level))
    completed: list[str] = []
    try:
        for video_id in video_ids:
            try:
                run_ocr(video_id, db, ocr_service)
            except Exception as error:
                if _is_gpu_oom(error):
                    return WorkerResult(
                        kind="enrichment",
                        completed_video_ids=tuple(completed),
                        oom_video_id=video_id,
                        oom_message=str(error),
                        requested_video_ids=video_ids,
                    )
                raise
            completed.append(video_id)
    finally:
        _close_resource(ocr_service)
        db.engine.dispose()
    return WorkerResult(
        kind="enrichment",
        completed_video_ids=tuple(completed),
        requested_video_ids=video_ids,
    )


def _ocr_config_for_retry(retry_level: int) -> OCRConfig:
    if retry_level < 0:
        raise ValueError("ocr_retry_level must not be negative.")
    base = OCRConfig(device="gpu:0")
    divisor = 2**retry_level
    return replace(
        base,
        detection_batch_size=max(1, base.detection_batch_size // divisor),
        recognition_batch_size=max(1, base.recognition_batch_size // divisor),
    )


def _is_gpu_oom(error: BaseException) -> bool:
    message = str(error).lower()
    markers = (
        "out of memory",
        "cuda error: memory",
        "cuda out of memory",
        "resourceexhausted",
        "resource exhausted",
    )
    return any(marker in message for marker in markers)


def _close_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


__all__ = [
    "GPUPreflight",
    "GPUPreflightError",
    "WorkerResult",
    "check_parallel_gpu_preflight",
    "run_parallel_workers",
    "run_worker_exclusively",
]
