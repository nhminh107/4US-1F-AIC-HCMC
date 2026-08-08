"""Batch and programmatic entrypoints for audio preprocessing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from BackEnd.app.contracts.pipeline import ShotMetadata, VideoMetadata

from . import extractor, normalizer, utils, vad
from .exporter import save_audio_segments_json
from .schemas import AudioSegment

logger = logging.getLogger(__name__)


def segment_shot(
    raw_audio_path: Path,
    start_ms: int,
    end_ms: int,
    output_path: Path,
) -> Path:
    """Cut exactly one shot into an intermediate WAV without padding."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = end_ms - start_ms
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_audio_path),
        "-ss",
        f"{start_ms / 1000:.6f}",
        "-t",
        f"{duration_ms / 1000:.6f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = utils.run_command(command)
    if result.returncode != 0 or not utils.validate_wav(output_path):
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        utils.cleanup_file(output_path)
        raise utils.AudioSegmentationError(
            f"FFmpeg shot segmentation failed: {stderr}"
        )
    return output_path


def preprocess_shot(
    video: VideoMetadata,
    shot: ShotMetadata,
    raw_audio_path: Path,
    output_dir: Path,
    language_hint: str | None = None,
) -> AudioSegment:
    """Process one already-validated shot using an existing raw WAV."""

    intermediate_path = utils.intermediate_audio_path(
        output_dir, video.video_id, shot.shot_index
    )
    normalized_path = utils.normalized_audio_path(
        output_dir, video.video_id, shot.shot_index
    )

    utils.cleanup_file(intermediate_path)
    utils.cleanup_file(normalized_path)

    try:
        segment_shot(raw_audio_path, shot.start_ms, shot.end_ms, intermediate_path)
        normalized_wav = normalizer.normalize_audio(intermediate_path, normalized_path)
        has_speech = vad.detect_speech(normalized_wav)
        return AudioSegment(
            segment_id=utils.segment_id(video.video_id, shot.shot_index),
            video_id=video.video_id,
            shot_id=shot.shot_id,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            audio_path=normalized_wav.resolve(),
            sample_rate=utils.SAMPLE_RATE,
            has_speech=has_speech,
            language_hint=language_hint,
        )
    except Exception:
        utils.cleanup_file(intermediate_path)
        utils.cleanup_file(normalized_path)
        raise
    finally:
        utils.cleanup_file(intermediate_path)


def preprocess_video(
    video: VideoMetadata,
    shots: list[ShotMetadata],
    output_dir: Path,
    language_hint: str | None = None,
) -> list[AudioSegment]:
    """Process all valid audio shots for one video."""

    if not extractor.has_audio_stream(video.video_path):
        logger.warning("Video has no audio stream: %s", video.video_id)
        return []

    raw_wav = extractor.get_or_extract_raw_audio(video, output_dir)
    try:
        audio_duration_ms = extractor.get_duration_ms(raw_wav)
        utils.validate_shots(shots, audio_duration_ms)

        results: list[AudioSegment] = []
        for shot in shots:
            try:
                results.append(
                    preprocess_shot(
                        video,
                        shot,
                        raw_wav,
                        output_dir,
                        language_hint=language_hint,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Shot audio preprocessing failed: video_id=%s shot_id=%s "
                    "shot_index=%s start_ms=%s end_ms=%s error=%s",
                    video.video_id,
                    shot.shot_id,
                    shot.shot_index,
                    shot.start_ms,
                    shot.end_ms,
                    exc,
                )
                continue

        save_audio_segments_json(results, output_dir, video.video_id)
        logger.info(
            "Audio preprocessing summary: video_id=%s requested=%s processed=%s "
            "skipped=%s speech=%s",
            video.video_id,
            len(shots),
            len(results),
            len(shots) - len(results),
            sum(segment.has_speech for segment in results),
        )
        return results
    finally:
        utils.cleanup_file(raw_wav)


def _load_shot_payload(shots_path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    with Path(shots_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        shots = payload.get("shots")
        if not isinstance(shots, list):
            raise ValueError("Shots JSON object must contain a 'shots' list.")
        video_id = payload.get("video_id")
        return (str(video_id) if video_id is not None else None), shots
    if isinstance(payload, list):
        return None, payload
    raise ValueError("Shots JSON must be either a list or an object with 'shots'.")


def _load_shots(shots_path: Path, default_video_id: str) -> tuple[str, list[ShotMetadata]]:
    payload_video_id, shot_payloads = _load_shot_payload(shots_path)
    video_id = payload_video_id or default_video_id
    shots: list[ShotMetadata] = []
    for index, item in enumerate(shot_payloads):
        if not isinstance(item, dict):
            raise ValueError("Every shot entry must be a JSON object.")
        shots.append(
            ShotMetadata(
                shot_id=str(item.get("shot_id", f"{video_id}_shot{index:04d}")),
                video_id=str(item.get("video_id", video_id)),
                shot_index=int(item.get("shot_index", index)),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                start_frame_idx=item.get("start_frame_idx"),
                end_frame_idx=item.get("end_frame_idx"),
            )
        )
    return video_id, shots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audio preprocessing.")
    parser.add_argument("--video", required=True, help="Path to a source video.")
    parser.add_argument("--shots", required=True, help="Path to shot metadata JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--language", default=None, help="Optional ASR language hint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    video_path = Path(args.video).expanduser().resolve()
    video_id, shots = _load_shots(Path(args.shots), video_path.stem)
    video = VideoMetadata(video_id=video_id, video_path=video_path)

    segments = preprocess_video(
        video,
        shots,
        Path(args.output_dir),
        language_hint=args.language,
    )
    print(f"Video: {video.video_id}")
    print(f"Shots requested: {len(shots)}")
    print(f"Shots processed: {len(segments)}")
    print(f"Shots skipped: {len(shots) - len(segments)}")
    print(f"Speech segments: {sum(segment.has_speech for segment in segments)}")
    print(f"Output directory: {utils.video_output_dir(Path(args.output_dir), video.video_id)}")


if __name__ == "__main__":
    main()
