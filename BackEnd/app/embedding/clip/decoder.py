"""Video frame decoder for clip embedding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from BackEnd.app.contracts.embedding import DecodedFrameBatch, VideoAsset
from BackEnd import CONFIG
from BackEnd.app.embedding.common.errors import DecodeError, MediaNotFoundError


class PyAVVideoDecoder:
    """Decode nearest RGB frames with PyAV."""

    def __init__(self, *, allow_full_scan_fallback: bool = True) -> None:
        self.allow_full_scan_fallback = bool(allow_full_scan_fallback)

    def decode_nearest_frames(
        self,
        video_asset: VideoAsset,
        timestamps_ms: list[int] | tuple[int, ...],
    ) -> DecodedFrameBatch:
        """Decode frames nearest to requested timestamps."""

        requested_timestamps = tuple(int(timestamp) for timestamp in timestamps_ms)
        if not video_asset.video_uri.is_file():
            return DecodedFrameBatch(
                video_id=video_asset.video_id,
                images=tuple(None for _ in requested_timestamps),
                requested_timestamps_ms=requested_timestamps,
                actual_timestamps_ms=tuple(None for _ in requested_timestamps),
                decode_statuses=tuple("media_not_found" for _ in requested_timestamps),
                metrics={
                    "open_count": 0,
                    "seek_count": 0,
                    "decoded_frame_count": 0,
                    "requested_frame_count": len(requested_timestamps),
                    "failed_frame_count": len(requested_timestamps),
                },
            )

        try:
            import av
        except ImportError as error:
            raise DecodeError("PyAV is not installed. Install the `av` package.") from error

        image_by_timestamp = {}
        actual_by_timestamp = {}
        status_by_timestamp = {
            timestamp: "decode_failed" for timestamp in set(requested_timestamps)
        }
        seek_count = 0
        decoded_frame_count = 0
        fallback_scan_count = 0
        try:
            with av.open(str(video_asset.video_uri)) as container:
                stream = container.streams.video[0]
                time_base = float(stream.time_base)
                for group in _group_timestamps(
                    sorted(set(requested_timestamps)),
                    CONFIG.DECODE_SEEK_GROUP_GAP_MS,
                ):
                    seek_count += 1
                    group_start_ms = max(0, group[0] - CONFIG.DECODE_TOLERANCE_MS)
                    group_end_ms = group[-1] + CONFIG.DECODE_TOLERANCE_MS
                    try:
                        target_pts = int((group_start_ms / 1000.0) / time_base)
                        container.seek(target_pts, any_frame=False, backward=True, stream=stream)
                        selected, decoded_count = _select_nearest_frames(
                            _iter_timestamped_frames(container.decode(stream), stream),
                            group,
                            start_ms=group_start_ms,
                            end_ms=group_end_ms,
                        )
                        decoded_frame_count += decoded_count
                        for requested_ms, (actual_ms, image) in selected.items():
                            image_by_timestamp[requested_ms] = image
                            actual_by_timestamp[requested_ms] = actual_ms
                            status_by_timestamp[requested_ms] = "success"
                    except Exception:
                        if not self.allow_full_scan_fallback:
                            continue
                        fallback_scan_count += 1
                        try:
                            container.seek(0, any_frame=False, stream=stream)
                            selected, decoded_count = _select_nearest_frames(
                                _iter_timestamped_frames(
                                    container.decode(stream),
                                    stream,
                                ),
                                group,
                                start_ms=group_start_ms,
                                end_ms=group_end_ms,
                            )
                            decoded_frame_count += decoded_count
                            for requested_ms, (actual_ms, image) in selected.items():
                                image_by_timestamp[requested_ms] = image
                                actual_by_timestamp[requested_ms] = actual_ms
                                status_by_timestamp[requested_ms] = "success"
                        except Exception:
                            continue
        except FileNotFoundError as error:
            raise MediaNotFoundError(str(video_asset.video_uri)) from error
        except Exception as error:
            raise DecodeError(f"Cannot decode video {video_asset.video_id}.") from error

        images = tuple(image_by_timestamp.get(timestamp) for timestamp in requested_timestamps)
        actual_timestamps = tuple(actual_by_timestamp.get(timestamp) for timestamp in requested_timestamps)
        statuses = tuple(status_by_timestamp.get(timestamp, "decode_failed") for timestamp in requested_timestamps)
        return DecodedFrameBatch(
            video_id=video_asset.video_id,
            images=images,
            requested_timestamps_ms=requested_timestamps,
            actual_timestamps_ms=actual_timestamps,
            decode_statuses=statuses,
            metrics={
                "open_count": 1,
                "seek_count": seek_count,
                "fallback_scan_count": fallback_scan_count,
                "decoded_frame_count": decoded_frame_count,
                "requested_frame_count": len(requested_timestamps),
                "failed_frame_count": sum(status != "success" for status in statuses),
            },
        )


def _iter_timestamped_frames(frames: Iterable[Any], stream: Any):
    """Yield decoded frames with their millisecond timestamps."""

    for frame in frames:
        if frame.pts is None:
            continue
        actual_ms = int(round(float(frame.pts * stream.time_base) * 1_000))
        yield actual_ms, frame


def _select_nearest_frames(
    frames: Iterable[tuple[int, Any]],
    requested_timestamps_ms: list[int] | tuple[int, ...],
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[dict[int, tuple[int, Any]], int]:
    """Select nearest decoded frames without materializing intermediate RGB images."""

    requested = tuple(sorted(set(int(value) for value in requested_timestamps_ms)))
    selected: dict[int, tuple[int, Any]] = {}
    image_cache: dict[int, Any] = {}
    previous: tuple[int, Any] | None = None
    requested_index = 0
    decoded_count = 0

    def select(requested_ms: int, candidate: tuple[int, Any]) -> None:
        actual_ms, frame = candidate
        image = image_cache.get(actual_ms)
        if image is None:
            image = frame.to_ndarray(format="rgb24")
            image_cache[actual_ms] = image
        selected[requested_ms] = (actual_ms, image)

    for actual_ms, frame in frames:
        if actual_ms < start_ms:
            continue
        if actual_ms > end_ms:
            break
        decoded_count += 1
        current = (actual_ms, frame)
        while requested_index < len(requested) and actual_ms >= requested[requested_index]:
            requested_ms = requested[requested_index]
            candidate = current
            if previous is not None and abs(previous[0] - requested_ms) <= abs(
                actual_ms - requested_ms
            ):
                candidate = previous
            select(requested_ms, candidate)
            requested_index += 1
        previous = current

    if previous is not None:
        while requested_index < len(requested):
            select(requested[requested_index], previous)
            requested_index += 1

    return selected, decoded_count


def _group_timestamps(timestamps_ms: list[int], max_gap_ms: int) -> list[list[int]]:
    if not timestamps_ms:
        return []
    groups = [[timestamps_ms[0]]]
    for timestamp in timestamps_ms[1:]:
        if timestamp - groups[-1][-1] <= max_gap_ms:
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])
    return groups
