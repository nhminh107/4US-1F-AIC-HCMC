"""Run tracking for one video already registered with PostgreSQL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from BackEnd.app.contracts.pipeline import VideoMetadata
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.pipeline.tracking import track_video
from BackEnd.app.tracking.tracking import ByteTrackService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track one video and print the persisted track summaries as JSON."
    )
    parser.add_argument("video_path", type=Path, help="Path to the source video")
    parser.add_argument(
        "--video-id",
        help="Video ID already present in PostgreSQL (default: video file stem)",
    )
    args = parser.parse_args()

    video_path = args.video_path.resolve()
    if not video_path.is_file():
        parser.error(f"Video does not exist: {video_path}")

    video = VideoMetadata(
        video_id=args.video_id or video_path.stem,
        video_path=video_path,
    )
    tracks = track_video(video, PostgreManager(), ByteTrackService())
    print(json.dumps([asdict(track) for track in tracks], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
