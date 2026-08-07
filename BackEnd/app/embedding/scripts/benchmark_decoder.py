"""Benchmark PyAV decoding metrics for one local video."""

from __future__ import annotations

import argparse
from pathlib import Path

from BackEnd.app.contracts.embedding import VideoAsset
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder


def benchmark_decoder(
    video_path: Path,
    *,
    video_id: str = "benchmark-video",
    timestamps_ms: tuple[int, ...] = (500, 1000, 1500, 2000),
) -> dict[str, int | float]:
    """Decode requested timestamps and return decoder metrics."""

    decoder = PyAVVideoDecoder()
    batch = decoder.decode_nearest_frames(
        VideoAsset(video_id=video_id, video_uri=video_path),
        timestamps_ms,
    )
    timestamp_errors = [
        abs(actual - requested)
        for requested, actual, status in zip(
            batch.requested_timestamps_ms,
            batch.actual_timestamps_ms,
            batch.decode_statuses,
        )
        if status == "success" and actual is not None
    ]
    metrics = dict(batch.metrics)
    metrics["timestamp_error_max_ms"] = max(timestamp_errors, default=0)
    metrics["timestamp_error_mean_ms"] = (
        sum(timestamp_errors) / len(timestamp_errors) if timestamp_errors else 0
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyAV decoder metrics.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", default="benchmark-video")
    parser.add_argument(
        "--timestamps-ms",
        default="500,1000,1500,2000",
        help="Comma-separated requested timestamps in milliseconds.",
    )
    args = parser.parse_args()

    timestamps = tuple(
        int(value.strip())
        for value in args.timestamps_ms.split(",")
        if value.strip()
    )
    metrics = benchmark_decoder(args.video, video_id=args.video_id, timestamps_ms=timestamps)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

