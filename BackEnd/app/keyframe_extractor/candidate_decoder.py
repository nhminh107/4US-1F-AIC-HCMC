"""In-memory candidate frame decoding for hybrid keyframe selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from BackEnd.app.contracts.embedding import VideoAsset
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder
from BackEnd.app.embedding.common.interfaces import VideoDecoder


class CandidateDecodeError(RuntimeError):
    """Raised when candidate frames cannot be decoded for semantic scoring."""


class PyAVCandidateFrameDecoder:
    """Decode sparse candidate frames in memory using the shared PyAV decoder."""

    def __init__(self, decoder: VideoDecoder | None = None) -> None:
        self.decoder = decoder or PyAVVideoDecoder(allow_full_scan_fallback=False)

    def decode(
        self,
        *,
        video_id: str,
        video_path: str | Path,
        frame_indices: Sequence[int],
        fps: float,
        session: object | None = None,
    ) -> dict[int, Image.Image]:
        """Return decoded RGB PIL images keyed by requested frame index."""

        if fps <= 0:
            raise ValueError("fps must be positive.")
        requested_frames = [int(frame_idx) for frame_idx in frame_indices]
        if not requested_frames:
            return {}

        timestamps_ms = [round(frame_idx / fps * 1000) for frame_idx in requested_frames]

        if session is not None and hasattr(session, "decode_nearest_frames"):
            batch = session.decode_nearest_frames(timestamps_ms)
        else:
            batch = self.decoder.decode_nearest_frames(
                VideoAsset(
                    video_id=video_id,
                    video_uri=Path(video_path),
                    nominal_fps=fps,
                ),
                timestamps_ms,
            )

        decoded: dict[int, Image.Image] = {}
        failures: list[str] = []
        for frame_idx, image, status in zip(requested_frames, batch.images, batch.decode_statuses):
            if status != "success" or image is None:
                failures.append(f"{frame_idx}:{status}")
                continue
            decoded[frame_idx] = _to_rgb_image(image)

        if failures:
            raise CandidateDecodeError(
                f"Cannot decode {len(failures)}/{len(requested_frames)} candidate frame(s) "
                f"for video '{video_id}' at '{video_path}': " + ", ".join(failures[:8])
            )
        return decoded


def _to_rgb_image(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise CandidateDecodeError("Decoded frame must have shape (height, width, 3).")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")
