"""Small command-line demo for the Clip Extractor module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .clip_extractor import ClipExtractor, ClipExtractorConfig
from .exceptions import ClipExtractorError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split one Shot record into Clip records."
    )
    parser.add_argument("shot_json", type=Path, help="JSON file containing one Shot")
    parser.add_argument(
        "--max-duration-ms",
        type=int,
        default=10_000,
        help="Maximum Clip duration (default: 10000)",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Create real MP4 files with FFmpeg",
    )
    parser.add_argument("--video-path", type=Path, help="Override Shot.video_path")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/clips"),
        help="Root directory for generated MP4 files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.shot_json.open("r", encoding="utf-8") as source:
            shot: Dict[str, Any] = json.load(source)

        extractor = ClipExtractor(
            ClipExtractorConfig(
                split_threshold_ms=args.max_duration_ms,
                max_clip_duration_ms=args.max_duration_ms,
                materialize_files=args.materialize,
                output_root=args.output_root,
            )
        )
        clips = extractor.run(shot, video_path=args.video_path)
    except (OSError, json.JSONDecodeError, ClipExtractorError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(clips, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
