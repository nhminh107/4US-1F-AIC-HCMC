"""JSON export for audio preprocessing artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import AudioSegment
from .utils import video_output_dir


def _segment_to_json(segment: AudioSegment) -> dict[str, object]:
    payload = asdict(segment)
    payload["audio_path"] = str(segment.audio_path.resolve())
    return payload


def save_audio_segments_json(
    segments: list[AudioSegment],
    output_dir: Path,
    video_id: str,
) -> Path:
    target_dir = video_output_dir(output_dir, video_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "audio_segments.json"
    payload = {
        "video_id": video_id,
        "segments": [_segment_to_json(segment) for segment in segments],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path


save_json = save_audio_segments_json
