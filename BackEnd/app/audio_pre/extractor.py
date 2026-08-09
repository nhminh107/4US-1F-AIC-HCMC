"""Full-video audio extraction helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from BackEnd.app.contracts.pipeline import VideoMetadata

from . import utils

logger = logging.getLogger(__name__)


def has_audio_stream(video_path: Path) -> bool:
    """Return whether a readable media file contains at least one audio stream."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    result = utils.run_command(command, capture_stdout=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise utils.AudioExtractionError(
            f"ffprobe failed while checking audio stream for {video_path}: {stderr}"
        )

    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise utils.AudioExtractionError(
            f"ffprobe returned invalid JSON for {video_path}."
        ) from exc

    return bool(payload.get("streams"))


def extract_full_audio(video: VideoMetadata, output_dir: Path) -> Path:
    """Extract the full audio track into a temporary raw WAV."""

    target_dir = utils.ensure_video_output_dir(output_dir, video.video_id)
    raw_path = target_dir / f"{video.video_id}_raw.wav"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video.video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(raw_path),
    ]

    logger.info("Extracting full-video audio: video_id=%s", video.video_id)
    result = utils.run_command(command)
    if result.returncode != 0 or not utils.validate_wav(raw_path):
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        utils.cleanup_file(raw_path)
        raise utils.AudioExtractionError(
            f"FFmpeg audio extraction failed for {video.video_id}: {stderr}"
        )

    logger.info("Extracted raw WAV: %s", raw_path)
    return raw_path


def get_or_extract_raw_audio(video: VideoMetadata, output_dir: Path) -> Path:
    """Reuse a valid raw WAV or re-extract it from the source video."""

    raw_path = utils.raw_audio_path(output_dir, video.video_id)
    if raw_path.exists():
        if utils.validate_wav(raw_path):
            logger.info("Reusing valid raw WAV: %s", raw_path)
            return raw_path
        logger.info("Discarding invalid raw WAV before re-extraction: %s", raw_path)
        utils.cleanup_file(raw_path)

    return extract_full_audio(video, output_dir)


def get_duration_ms(raw_audio_path: Path) -> float:
    return utils.get_wav_duration_ms(raw_audio_path)
