"""Persistent PyAV video container session for fast candidate decoding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from BackEnd.app.contracts.embedding import DecodedFrameBatch
from BackEnd.app.embedding import CONFIG
from BackEnd.app.embedding.common.errors import DecodeError


class PyAVVideoSession:
    """Session container giữ duy nhất 1 av.open() mở trong suốt chu kỳ làm việc."""

    def __init__(self, video_path: str | Path, nominal_fps: float) -> None:
        self.video_path = Path(video_path)
        self.nominal_fps = float(nominal_fps)
        self._container: Any | None = None
        self._stream: Any | None = None
        self._cache: dict[int, Any] = {}

    def prefetch_timestamps(self, timestamps_ms: Sequence[int]) -> int:
        """Prefetch và cache toàn bộ candidate timestamps của video trong 1 lần lướt single-pass."""

        uncached = [int(ts) for ts in set(timestamps_ms) if int(ts) not in self._cache]
        if not uncached:
            return 0
        decoded = self.decode_all_timestamps_sequential(uncached)
        self._cache.update(decoded)
        return len(decoded)

    def __enter__(self) -> PyAVVideoSession:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._container is not None:
            return
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found at '{self.video_path}'")

        try:
            import av
        except ImportError as error:
            raise DecodeError("PyAV is not installed. Install the `av` package.") from error

        try:
            self._container = av.open(str(self.video_path))
            self._stream = self._container.streams.video[0]
        except Exception as error:
            self.close()
            raise DecodeError(f"Failed to open video container '{self.video_path}': {error}") from error

    def close(self) -> None:
        # Candidate RGB arrays can be large. Sessions are scoped to one video,
        # so retaining this cache after close only creates avoidable RAM peaks.
        self._cache.clear()
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            finally:
                self._container = None
                self._stream = None

    def decode_nearest_frames(self, timestamps_ms: Sequence[int]) -> DecodedFrameBatch:
        """Decode danh sách timestamps từ container đang mở."""

        self._ensure_open()
        requested_timestamps = tuple(int(ts) for ts in timestamps_ms)
        if not requested_timestamps:
            return DecodedFrameBatch(
                video_id=self.video_path.stem,
                images=(),
                requested_timestamps_ms=(),
                actual_timestamps_ms=(),
                decode_statuses=(),
                metrics={"open_count": 0, "seek_count": 0},
            )

        assert self._container is not None
        assert self._stream is not None

        time_base = float(self._stream.time_base)
        image_by_timestamp: dict[int, Any] = {}
        actual_by_timestamp: dict[int, int] = {}
        status_by_timestamp: dict[int, str] = {
            ts: "decode_failed" for ts in set(requested_timestamps)
        }

        # Kiểm tra RAM cache trước
        uncached_timestamps: list[int] = []
        for ts in set(requested_timestamps):
            if ts in self._cache:
                image_by_timestamp[ts] = self._cache[ts]
                actual_by_timestamp[ts] = ts
                status_by_timestamp[ts] = "success"
            else:
                uncached_timestamps.append(ts)

        if not uncached_timestamps:
            images = tuple(image_by_timestamp.get(ts) for ts in requested_timestamps)
            actuals = tuple(actual_by_timestamp.get(ts) for ts in requested_timestamps)
            statuses = tuple(status_by_timestamp.get(ts, "decode_failed") for ts in requested_timestamps)

            return DecodedFrameBatch(
                video_id=self.video_path.stem,
                images=images,
                requested_timestamps_ms=requested_timestamps,
                actual_timestamps_ms=actuals,
                decode_statuses=statuses,
                metrics={"open_count": 0, "seek_count": 0},
            )

        # Tìm nhóm timestamps chưa được cache
        sorted_unique = sorted(uncached_timestamps)
        group_gap = int(CONFIG.DECODE_SEEK_GROUP_GAP_MS)
        tolerance = int(CONFIG.DECODE_TOLERANCE_MS)

        groups: list[list[int]] = []
        for ts in sorted_unique:
            if not groups or ts - groups[-1][-1] > group_gap:
                groups.append([ts])
            else:
                groups[-1].append(ts)

        for group in groups:
            group_start_ms = max(0, group[0] - tolerance)
            group_end_ms = group[-1] + tolerance
            target_pts = int((group_start_ms / 1000.0) / time_base)

            try:
                self._container.seek(target_pts, any_frame=False, backward=True, stream=self._stream)
                candidates: list[tuple[int, Any]] = []
                for frame in self._container.decode(self._stream):
                    if frame.pts is None:
                        continue
                    actual_ms = int(round(float(frame.pts * self._stream.time_base) * 1000))
                    if actual_ms > group_end_ms:
                        break
                    if actual_ms >= group_start_ms:
                        candidates.append((actual_ms, frame.to_ndarray(format="rgb24")))

                for req_ms in group:
                    if not candidates:
                        continue
                    best_actual, best_img = min(candidates, key=lambda item: abs(item[0] - req_ms))
                    image_by_timestamp[req_ms] = best_img
                    actual_by_timestamp[req_ms] = best_actual
                    status_by_timestamp[req_ms] = "success"
            except Exception:
                continue

        images = tuple(image_by_timestamp.get(ts) for ts in requested_timestamps)
        actuals = tuple(actual_by_timestamp.get(ts) for ts in requested_timestamps)
        statuses = tuple(status_by_timestamp.get(ts, "decode_failed") for ts in requested_timestamps)

        return DecodedFrameBatch(
            video_id=self.video_path.stem,
            images=images,
            requested_timestamps_ms=requested_timestamps,
            actual_timestamps_ms=actuals,
            decode_statuses=statuses,
            metrics={"open_count": 1, "seek_count": len(groups)},
        )

    def decode_all_timestamps_sequential(
        self,
        all_timestamps_ms: Sequence[int],
    ) -> dict[int, Any]:
        """Giải mã TẤT CẢ các timestamps của TOÀN BỘ VIDEO trong ĐÚNG 1 LẦN PASS LƯỚT XUÔI SEQUENTIAL."""

        self._ensure_open()
        assert self._container is not None
        assert self._stream is not None

        requested_set = set(int(ts) for ts in all_timestamps_ms)
        if not requested_set:
            return {}

        sorted_targets = sorted(requested_set)
        time_base = float(self._stream.time_base)
        tolerance = int(CONFIG.DECODE_TOLERANCE_MS)

        results: dict[int, Any] = {}
        target_idx = 0
        num_targets = len(sorted_targets)

        # Trở về đầu stream nếu cần
        try:
            self._container.seek(0, any_frame=False, stream=self._stream)
        except Exception:
            pass

        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            current_ms = int(round(float(frame.pts * time_base) * 1000))

            while target_idx < num_targets:
                target_ms = sorted_targets[target_idx]
                if current_ms < target_ms - tolerance:
                    break
                if abs(current_ms - target_ms) <= tolerance or current_ms >= target_ms:
                    results[target_ms] = frame.to_ndarray(format="rgb24")
                    target_idx += 1
                else:
                    target_idx += 1

            if target_idx >= num_targets and current_ms > sorted_targets[-1] + tolerance:
                break

        return results
