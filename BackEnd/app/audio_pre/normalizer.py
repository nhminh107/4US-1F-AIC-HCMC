"""Shot-level WAV normalization."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from . import utils

logger = logging.getLogger(__name__)


def _load_as_float32_mono_16k(input_path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(utils.CHANNELS),
        "-ar",
        str(utils.SAMPLE_RATE),
        "-",
    ]
    result = utils.run_command(command, capture_stdout=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise utils.AudioNormalizationError(
            f"FFmpeg format conversion failed for {input_path}: {stderr}"
        )
    if not result.stdout:
        raise utils.AudioNormalizationError(f"No audio samples decoded: {input_path}")
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    """Resample, downmix, peak-normalize, and write PCM16 WAV."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_as_float32_mono_16k(input_path)
    if samples.size == 0:
        raise utils.AudioNormalizationError(f"No audio samples decoded: {input_path}")

    peak = float(np.max(np.abs(samples)))
    if peak > 0.0:
        samples = samples / peak
    samples = np.clip(samples, -1.0, 1.0)

    pcm = np.rint(samples * 32767.0).astype("<i2", copy=False)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(utils.CHANNELS)
        wav_file.setsampwidth(utils.SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(utils.SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    if not utils.validate_normalized_wav(output_path):
        utils.cleanup_file(output_path)
        raise utils.AudioNormalizationError(
            f"Normalized WAV failed validation: {output_path}"
        )

    logger.info("Normalized WAV written: %s", output_path)
    return output_path.resolve()


normalize = normalize_audio
