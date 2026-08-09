"""WebRTC VAD speech detection for normalized WAV files."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from . import utils

logger = logging.getLogger(__name__)


def _create_vad(mode: int):
    try:
        import webrtcvad
    except ImportError as exc:
        raise utils.AudioVADError(
            "webrtcvad is required for audio preprocessing VAD."
        ) from exc
    return webrtcvad.Vad(mode)


def detect_speech(
    audio_path: Path,
    *,
    mode: int = utils.VAD_MODE,
    frame_duration_ms: int = utils.VAD_FRAME_DURATION_MS,
) -> bool:
    """Return True when any complete VAD frame contains speech."""

    if frame_duration_ms not in {10, 20, 30}:
        raise ValueError("WebRTC VAD frame duration must be 10, 20, or 30 ms.")

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error, OSError) as exc:
        raise utils.AudioVADError(f"Cannot read normalized WAV: {audio_path}") from exc

    if (
        sample_rate != utils.SAMPLE_RATE
        or channels != utils.CHANNELS
        or sample_width != utils.SAMPLE_WIDTH_BYTES
    ):
        raise utils.AudioVADError(
            "VAD input must be 16 kHz, mono, 16-bit PCM WAV."
        )

    vad = _create_vad(mode)
    frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width
    for offset in range(0, len(frames) - frame_size + 1, frame_size):
        if vad.is_speech(frames[offset : offset + frame_size], sample_rate):
            return True
    return False


detect = detect_speech
