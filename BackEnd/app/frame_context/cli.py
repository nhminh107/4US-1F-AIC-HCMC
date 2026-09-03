"""Command-line entry point for building FrameContext V1."""

from __future__ import annotations

import argparse
from pathlib import Path

from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.frame_context.artifact import write_frame_context_artifact
from BackEnd.app.frame_context.builder import build_frame_context
from BackEnd.app.frame_context.postgres_source import load_frame_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FrameContext V1 artifact.")
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/context"))
    parser.add_argument("--database-url")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--minimum-object-confidence", type=float, default=0.25)
    args = parser.parse_args()

    manager = PostgreManager(database_url=args.database_url)
    try:
        evidence = load_frame_evidence(
            manager.engine,
            video_ids=args.video_ids,
            minimum_object_confidence=args.minimum_object_confidence,
        )
        records = [build_frame_context(item) for item in evidence]
        artifact_root = write_frame_context_artifact(
            records,
            args.output_root,
            build_id=args.build_id,
        )
    finally:
        manager.engine.dispose()

    searchable = sum(bool(record.context_text) for record in records)
    print(f"Wrote {len(records)} contexts ({searchable} searchable) to {artifact_root}")


if __name__ == "__main__":
    main()
