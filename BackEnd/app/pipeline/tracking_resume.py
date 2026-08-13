"""Resume tracking from a known video boundary using independent GPU workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Sequence


def _split_round_robin(video_ids: Sequence[str], worker_count: int) -> list[tuple[str, ...]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    return [tuple(video_ids[index::worker_count]) for index in range(worker_count)]


def _run_shard(worker_index: int, video_ids: tuple[str, ...]) -> dict[str, object]:
    """Track one non-overlapping shard and commit each completed video."""

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.pipeline.tracking import track_video
    from BackEnd.app.tracking.tracking import YOLOTrackingService

    db = PostgreManager()
    tracker = YOLOTrackingService()
    completed: list[str] = []
    try:
        videos = {video.video_id: video for video in db.get_list_video()}
        for position, video_id in enumerate(video_ids, start=1):
            track_video(videos[video_id], db, tracker)
            completed.append(video_id)
            print(
                f"[tracking-resume] worker={worker_index} "
                f"{position}/{len(video_ids)} committed {video_id}",
                flush=True,
            )
    finally:
        close = getattr(tracker, "close", None)
        if callable(close):
            close()
        db.engine.dispose()
    return {
        "worker_index": worker_index,
        "completed": len(completed),
        "first_video_id": video_ids[0] if video_ids else None,
        "last_video_id": video_ids[-1] if video_ids else None,
    }


def run_resume(*, start_video_id: str, worker_count: int) -> list[dict[str, object]]:
    """Run tracking for all videos from ``start_video_id`` inclusively."""

    from BackEnd.app.database.postgre_db import PostgreManager
    from BackEnd.app.database.models import ObjectTrack, Shot
    from sqlalchemy import select

    db = PostgreManager()
    try:
        with db.session_factory() as session:
            completed_video_ids = set(
                session.scalars(
                    select(Shot.video_id)
                    .join(ObjectTrack, ObjectTrack.shot_id == Shot.shot_id)
                    .distinct()
                )
            )
        video_ids = [
            video.video_id
            for video in db.get_list_video()
            if (
                video.video_id >= start_video_id
                and video.video_id not in completed_video_ids
            )
        ]
    finally:
        db.engine.dispose()
    if not video_ids:
        print(f"[tracking-resume] no untracked videos from {start_video_id}")
        return []

    shards = [shard for shard in _split_round_robin(video_ids, worker_count) if shard]
    print(
        f"[tracking-resume] start={start_video_id}, pending={len(video_ids)}, "
        f"workers={len(shards)}",
        flush=True,
    )
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as executor:
        futures = [
            executor.submit(_run_shard, worker_index, shard)
            for worker_index, shard in enumerate(shards, start=1)
        ]
        return [future.result() for future in futures]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume tracking from a known uncommitted video boundary."
    )
    parser.add_argument("--start-video-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.workers <= 0:
        raise ValueError("--workers must be positive.")
    if arguments.dry_run:
        from BackEnd.app.database.postgre_db import PostgreManager

        db = PostgreManager()
        try:
            video_ids = [
                video.video_id
                for video in db.get_list_video()
                if video.video_id >= arguments.start_video_id
            ]
        finally:
            db.engine.dispose()
        if not video_ids or video_ids[0] != arguments.start_video_id:
            raise ValueError(f"Unknown start video ID: {arguments.start_video_id}.")
        shards = _split_round_robin(video_ids, arguments.workers)
        for worker_index, shard in enumerate(shards, start=1):
            print(
                f"worker={worker_index}, videos={len(shard)}, "
                f"first={shard[0] if shard else None}, last={shard[-1] if shard else None}"
            )
        return

    summaries = run_resume(
        start_video_id=arguments.start_video_id,
        worker_count=arguments.workers,
    )
    print(f"[tracking-resume] completed: {summaries}", flush=True)


if __name__ == "__main__":
    main()
