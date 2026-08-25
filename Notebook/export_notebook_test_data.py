"""Export notebook test inputs from PostgreSQL without modifying database data."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from sqlalchemy import select

from BackEnd.app.database.models import Frame, Shot
from BackEnd.app.database.postgre_db import PostgreManager


VIDEO_COLUMNS = ("video_id", "video_url")
SHOT_COLUMNS = (
    "shot_id",
    "video_id",
    "shot_index",
    "start_ms",
    "end_ms",
    "start_frame_idx",
    "end_frame_idx",
)
FRAME_COLUMNS = (
    "frame_id",
    "video_id",
    "shot_id",
    "timestamp_ms",
    "fps",
    "frame_idx",
    "source",
    "n",
    "pts_time",
    "frame_path",
    "image_url",
    "width",
    "height",
)

PUBLIC_IMAGE_URL_ENV_NAMES = (
    "R2_PUBLIC_URL",
    "R2_PUBLIC_BASE_URL",
    "IMAGE_PUBLIC_URL",
    "CLOUDFLARE_R2_PUBLIC_URL",
)


def read_video_manifest(path: Path) -> list[dict[str, str]]:
    """Read ``video_id,url`` rows, preserving their supplied order."""

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected video_id,url")
            video_id, video_url = (value.strip() for value in row)
            if not video_id or not video_url:
                raise ValueError(f"{path}:{line_number}: video_id and URL are required")
            if video_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate video_id {video_id!r}")
            seen.add(video_id)
            rows.append({"video_id": video_id, "video_url": video_url})
    if not rows:
        raise ValueError(f"{path}: no videos found")
    return rows


def normalize(value: Any) -> Any:
    """Use empty CSV cells for nullable database values."""

    return "" if value is None else value


def build_image_url(public_base_url: str | None, frame_path: str | None) -> str:
    """Build a public R2 image URL while retaining ``frame_path`` as the object key."""

    if not public_base_url or not frame_path:
        return ""
    if frame_path.startswith(("https://", "http://")):
        return frame_path
    return f"{public_base_url.rstrip('/')}/{quote(frame_path.lstrip('/'), safe='/')}"


def public_base_url(value: str | None) -> str | None:
    """Use the CLI value first, then a deliberately explicit public URL variable."""

    if value:
        return value.strip()
    for name in PUBLIC_IMAGE_URL_ENV_NAMES:
        if configured := os.environ.get(name, "").strip():
            return configured
    return None


def infer_public_base_url(video_rows: Iterable[dict[str, str]]) -> str | None:
    """Infer one R2 public origin from the supplied public video URLs."""

    origins = {
        f"{parsed.scheme}://{parsed.netloc}"
        for row in video_rows
        if (parsed := urlsplit(row["video_url"])).scheme in {"http", "https"}
        and parsed.netloc
    }
    return next(iter(origins)) if len(origins) == 1 else None


def write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def export_rows(
    video_rows: list[dict[str, str]],
    output_dir: Path,
    image_public_url: str | None = None,
) -> dict[str, int]:
    """Read shot/frame metadata and write notebook-compatible CSV files."""

    video_ids = [row["video_id"] for row in video_rows]
    order = {video_id: index for index, video_id in enumerate(video_ids)}
    database = PostgreManager()
    with database.session_factory() as session:
        shots = session.scalars(
            select(Shot)
            .where(Shot.video_id.in_(video_ids))
            .order_by(Shot.video_id, Shot.shot_index)
        ).all()
        frames = session.scalars(
            select(Frame)
            .where(Frame.video_id.in_(video_ids))
            .order_by(Frame.video_id, Frame.frame_idx, Frame.frame_id)
        ).all()

    shot_rows = [
        {column: normalize(getattr(shot, column)) for column in SHOT_COLUMNS}
        for shot in sorted(shots, key=lambda shot: (order[shot.video_id], shot.shot_index))
    ]
    base_url = public_base_url(image_public_url) or infer_public_base_url(video_rows)
    frame_rows = []
    for frame in sorted(
        frames,
        key=lambda frame: (order[frame.video_id], frame.frame_idx, frame.frame_id),
    ):
        row = {
            column: normalize(getattr(frame, column))
            for column in FRAME_COLUMNS
            if column != "image_url"
        }
        row["image_url"] = build_image_url(base_url, row["frame_path"])
        frame_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "videos.csv", VIDEO_COLUMNS, video_rows)
    for name in ("shot.csv", "shots.csv"):
        write_csv(output_dir / name, SHOT_COLUMNS, shot_rows)
    for name in ("keyframe.csv", "keyframes.csv"):
        write_csv(output_dir / name, FRAME_COLUMNS, frame_rows)

    return {"videos": len(video_rows), "shots": len(shot_rows), "keyframes": len(frame_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-file", type=Path, default=Path("Notebook/video.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("Notebook/test_data"))
    parser.add_argument(
        "--image-public-url",
        help=(
            "R2 public URL prefix. If omitted, reads one of "
            f"{', '.join(PUBLIC_IMAGE_URL_ENV_NAMES)}."
        ),
    )
    args = parser.parse_args()

    counts = export_rows(
        read_video_manifest(args.video_file),
        args.output_dir,
        image_public_url=args.image_public_url,
    )
    print(
        f"Exported {counts['videos']} videos, {counts['shots']} shots, "
        f"and {counts['keyframes']} keyframes to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
