"""Resolve local video assets for clip embedding."""

from __future__ import annotations

from pathlib import Path

from BackEnd.app.contracts.embedding import VideoAsset
from BackEnd.app.embedding.common.errors import MediaNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "video"


class VideoRepository:
    """Resolve video IDs to local media paths without hard-coded absolutes."""

    def __init__(self, video_dir: str | Path = DEFAULT_VIDEO_DIR) -> None:
        self.video_dir = Path(video_dir)
        self._resolved_cache: dict[str, Path] = {}

    def resolve_video(self, video_id: str) -> VideoAsset:
        """Return a local video asset for a canonical video ID."""

        if not video_id or not video_id.strip():
            raise ValueError("video_id must be a non-empty string.")

        if video_id in self._resolved_cache:
            cached_path = self._resolved_cache[video_id]
            if cached_path.is_file():
                return VideoAsset(video_id=video_id, video_uri=cached_path)

        candidates = [
            self.video_dir / f"{video_id}.mp4",
            self.video_dir / video_id / f"{video_id}.mp4",
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._resolved_cache[video_id] = candidate
                return VideoAsset(video_id=video_id, video_uri=candidate)

        # Smart Fallback: Search in batch subdirectories under dataset root (e.g. data/Videos_L21_a/video/L21_V001.mp4)
        parent_dir = self.video_dir.parent if self.video_dir.name == "video" else self.video_dir
        patterns = [
            f"Videos_*/video/{video_id}.mp4",
            f"*/video/{video_id}.mp4",
            f"**/{video_id}.mp4",
        ]
        for pattern in patterns:
            for match in parent_dir.glob(pattern):
                if match.is_file():
                    self._resolved_cache[video_id] = match
                    return VideoAsset(video_id=video_id, video_uri=match)

        raise MediaNotFoundError(f"Video media not found for video_id={video_id}.")


