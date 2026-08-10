"""Shared helpers for audio preprocessing."""

from __future__ import annotations

import logging
import os
import subprocess
import wave
from pathlib import Path
from typing import Sequence

from BackEnd.CONFIG import (
    AUDIO_CHANNELS as CHANNELS,
    AUDIO_SAMPLE_RATE as SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH_BYTES as SAMPLE_WIDTH_BYTES,
    AUDIO_VAD_FRAME_DURATION_MS as VAD_FRAME_DURATION_MS,
    AUDIO_VAD_MODE as VAD_MODE,
)
from BackEnd.app.contracts.pipeline import ShotMetadata

logger = logging.getLogger(__name__)

class AudioPreprocessingError(RuntimeError):
    """Base exception for audio preprocessing infrastructure failures."""


class AudioExtractionError(AudioPreprocessingError):
    """Raised when full-video audio extraction fails."""


class AudioSegmentationError(AudioPreprocessingError):
    """Raised when shot-level audio segmentation fails."""


class AudioNormalizationError(AudioPreprocessingError):
    """Raised when normalization fails."""


class AudioVADError(AudioPreprocessingError):
    """Raised when speech detection fails."""


def video_output_dir(output_dir: Path, video_id: str) -> Path:
    return Path(output_dir).expanduser().resolve() / video_id


def raw_audio_path(output_dir: Path, video_id: str) -> Path:
    return video_output_dir(output_dir, video_id) / f"{video_id}_raw.wav"


def segment_id(video_id: str, shot_index: int) -> str:
    return f"{video_id}_shot{shot_index:04d}"


def normalized_audio_path(output_dir: Path, video_id: str, shot_index: int) -> Path:
    return video_output_dir(output_dir, video_id) / f"{segment_id(video_id, shot_index)}.wav"


def intermediate_audio_path(output_dir: Path, video_id: str, shot_index: int) -> Path:
    return (
        video_output_dir(output_dir, video_id)
        / f"{segment_id(video_id, shot_index)}.raw.tmp.wav"
    )


def cleanup_file(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Failed to remove audio artifact %s: %s", path, exc)


def run_command(command: Sequence[str], *, capture_stdout: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def validate_wav(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or not os.access(path, os.R_OK):
        return False

    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getframerate() > 0
                and wav_file.getnchannels() > 0
                and wav_file.getsampwidth() > 0
                and wav_file.getnframes() > 0
            )
    except (EOFError, wave.Error, OSError):
        return False


def validate_normalized_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getframerate() == SAMPLE_RATE
                and wav_file.getnchannels() == CHANNELS
                and wav_file.getsampwidth() == SAMPLE_WIDTH_BYTES
                and wav_file.getnframes() > 0
            )
    except (EOFError, wave.Error, OSError):
        return False


def get_wav_duration_ms(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise ValueError(f"Invalid WAV frame rate: {path}")
            return wav_file.getnframes() / frame_rate * 1000.0
    except (EOFError, wave.Error, OSError) as exc:
        raise ValueError(f"Cannot determine WAV duration: {path}") from exc


def validate_shots(shots: list[ShotMetadata], audio_duration_ms: float) -> None:
    for shot in shots:
        if shot.start_ms < 0:
            raise ValueError(
                f"Invalid shot {shot.shot_id}: start_ms must be >= 0."
            )
        if shot.end_ms <= shot.start_ms:
            raise ValueError(
                f"Invalid shot {shot.shot_id}: end_ms must be greater than start_ms."
            )
        if shot.start_ms >= audio_duration_ms:
            raise ValueError(
                f"Invalid shot {shot.shot_id}: start_ms exceeds audio duration."
            )
        if shot.end_ms > audio_duration_ms:
            raise ValueError(
                f"Invalid shot {shot.shot_id}: end_ms exceeds audio duration."
            )


def ensure_video_output_dir(output_dir: Path, video_id: str) -> Path:
    target_dir = video_output_dir(output_dir, video_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir
