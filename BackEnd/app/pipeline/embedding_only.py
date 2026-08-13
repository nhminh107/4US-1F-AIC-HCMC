"""Run only the embedding stages with exclusive access to shared FAISS files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from typing import Callable, Iterator

from sqlalchemy import select

from BackEnd import CONFIG
from BackEnd.app.database.faiss_db import FAISS_Manager
from BackEnd.app.database.models import (
    ClipEmbeddingRecord,
    FrameEmbeddingRecord,
    ShotEmbeddingRecord,
)
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.pipeline.embedding import EmbeddingPipeline


def _process_is_stopped(pid: int) -> bool:
    """Return whether a Linux process exists in a stopped/traced state."""

    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False

    state = next((line for line in lines if line.startswith("State:")), "")
    return "T (stopped)" in state or "t (tracing stop)" in state


@contextmanager
def _exclusive_faiss_writer(faiss_dir: Path) -> Iterator[None]:
    """Prevent two embedding-only processes from writing shared indexes."""

    faiss_dir.mkdir(parents=True, exist_ok=True)
    lock_path = faiss_dir / ".embedding-writer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another embedding writer holds the FAISS lock: {lock_path}."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for GPU embedding.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run embedding on CPU.")
    print(f"Embedding device: {torch.cuda.get_device_name(0)}", flush=True)


def _validate_ocr_result(path: Path, expected_completed_videos: int) -> None:
    """Require a durable successful OCR result before embedding handoff."""

    if expected_completed_videos <= 0:
        raise ValueError("expected_completed_videos must be positive.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"OCR result does not exist: {path}.") from error
    if payload.get("error") is not None:
        raise RuntimeError(f"OCR did not complete successfully: {payload['error']}.")
    completed = payload.get("completed_video_ids")
    if not isinstance(completed, list) or len(completed) != expected_completed_videos:
        completed_count = len(completed) if isinstance(completed, list) else 0
        raise RuntimeError(
            "OCR completion count differs: "
            f"expected={expected_completed_videos}, actual={completed_count}."
        )
    if len(set(completed)) != len(completed):
        raise RuntimeError("OCR result contains duplicate completed video IDs.")
    print(
        f"[embedding] OCR handoff verified: {len(completed)} video(s)",
        flush=True,
    )


def _run_stage(
    *,
    stage_name: str,
    video_ids: list[str],
    callback: Callable[[str], list[object]],
) -> None:
    total = len(video_ids)
    print(f"[embedding] {stage_name} started: {total} video(s)", flush=True)
    for position, video_id in enumerate(video_ids, start=1):
        mappings = callback(video_id)
        print(
            f"[embedding] {stage_name} {position}/{total}: "
            f"{video_id}, mappings={len(mappings)}",
            flush=True,
        )
    print(f"[embedding] {stage_name} completed", flush=True)


def _validate_db_faiss_consistency(
    db: PostgreManager,
    faiss_manager: FAISS_Manager,
) -> None:
    """Ensure every selected DB mapping has a corresponding FAISS vector."""

    with db.session_factory() as session:
        frame_ids = list(
            session.scalars(
                select(FrameEmbeddingRecord.faiss_id).where(
                    FrameEmbeddingRecord.index_version == faiss_manager.version,
                    FrameEmbeddingRecord.model_name == faiss_manager.model_name,
                )
            )
        )
        clip_ids = list(
            session.scalars(
                select(ClipEmbeddingRecord.faiss_id).where(
                    ClipEmbeddingRecord.index_version == faiss_manager.version,
                    ClipEmbeddingRecord.model_name == faiss_manager.model_name,
                )
            )
        )
        shot_ids = list(
            session.scalars(
                select(ShotEmbeddingRecord.faiss_id).where(
                    ShotEmbeddingRecord.index_version == faiss_manager.version,
                    ShotEmbeddingRecord.model_name == faiss_manager.model_name,
                    ShotEmbeddingRecord.model_version == faiss_manager.model_version,
                )
            )
        )

    faiss_manager.validate_ids("frame", frame_ids)
    faiss_manager.validate_ids("clip", clip_ids)
    faiss_manager.validate_ids("shot", shot_ids)
    counts = {
        "frame": (len(frame_ids), faiss_manager.frame_idx.ntotal),
        "clip": (len(clip_ids), faiss_manager.clip_idx.ntotal),
        "shot": (len(shot_ids), faiss_manager.shot_idx.ntotal),
    }
    mismatches = {
        name: values for name, values in counts.items() if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(
            "DB/FAISS mapping counts differ (db, faiss): "
            f"{mismatches}."
        )
    print(f"[embedding] DB/FAISS consistency verified: {counts}", flush=True)


def run_embedding_only(
    *,
    faiss_dir: Path,
    stopped_orchestrator_pid: int | None,
    ocr_result_path: Path | None = None,
    expected_completed_videos: int | None = None,
) -> None:
    """Embed all videos while an older orchestrator remains safely stopped."""

    if stopped_orchestrator_pid is not None and not _process_is_stopped(
        stopped_orchestrator_pid
    ):
        raise RuntimeError(
            f"Orchestrator PID {stopped_orchestrator_pid} is not stopped."
        )
    if (ocr_result_path is None) != (expected_completed_videos is None):
        raise ValueError(
            "ocr_result_path and expected_completed_videos must be provided together."
        )
    if ocr_result_path is not None and expected_completed_videos is not None:
        _validate_ocr_result(ocr_result_path, expected_completed_videos)
    _require_cuda()

    resolved_faiss_dir = faiss_dir.expanduser().resolve()
    with _exclusive_faiss_writer(resolved_faiss_dir):
        db = PostgreManager()
        faiss_manager = FAISS_Manager(
            img_dim=CONFIG.CLIP_DIMENSION,
            clip_dim=CONFIG.CLIP_DIMENSION,
            shot_dim=CONFIG.CLIP_DIMENSION,
            model_name=CONFIG.CLIP_MODEL,
            data_path=resolved_faiss_dir,
            load_existing=True,
        )
        pipeline = EmbeddingPipeline(db=db, faiss_manager=faiss_manager)
        try:
            video_ids = [video.video_id for video in db.get_list_video()]
            _run_stage(
                stage_name="frames",
                video_ids=video_ids,
                callback=pipeline.embed_frames,
            )
            total = len(video_ids)
            print(f"[embedding] clip-shot started: {total} video(s)", flush=True)
            for position, video_id in enumerate(video_ids, start=1):
                try:
                    clip_mappings = pipeline.embed_clips(video_id)
                    shot_mappings = pipeline.embed_shot(video_id)
                finally:
                    pipeline.release_video_cache(video_id)
                print(
                    f"[embedding] clip-shot {position}/{total}: {video_id}, "
                    f"clips={len(clip_mappings)}, shots={len(shot_mappings)}",
                    flush=True,
                )
            print("[embedding] clip-shot completed", flush=True)
            _validate_db_faiss_consistency(db, faiss_manager)
        finally:
            pipeline.close()
            db.engine.dispose()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame, clip and shot embedding without other stages."
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("artifacts/faiss"),
    )
    parser.add_argument(
        "--stopped-orchestrator-pid",
        type=int,
        default=None,
        help="Refuse to start unless this PID is in a stopped state.",
    )
    parser.add_argument("--ocr-result-path", type=Path, default=None)
    parser.add_argument("--expected-completed-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_embedding_only(
        faiss_dir=arguments.faiss_dir,
        stopped_orchestrator_pid=arguments.stopped_orchestrator_pid,
        ocr_result_path=arguments.ocr_result_path,
        expected_completed_videos=arguments.expected_completed_videos,
    )


if __name__ == "__main__":
    main()
