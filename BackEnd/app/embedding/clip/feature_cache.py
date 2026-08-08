"""Small per-work-unit cache for decoded frame embeddings."""

from __future__ import annotations


class FrameFeatureCache:
    """Map `(video_id, timestamp_ms, embedding_space_id)` to one vector."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, int, str], object] = {}

    def get(self, video_id: str, timestamp_ms: int, embedding_space_id: str):
        return self._vectors.get((video_id, timestamp_ms, embedding_space_id))

    def set(self, video_id: str, timestamp_ms: int, embedding_space_id: str, vector) -> None:
        self._vectors[(video_id, timestamp_ms, embedding_space_id)] = vector

    def __len__(self) -> int:
        return len(self._vectors)

