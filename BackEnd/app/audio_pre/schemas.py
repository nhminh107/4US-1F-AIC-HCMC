"""Audio preprocessing data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """A normalized shot-level audio artifact ready for ASR."""

    segment_id: str
    video_id: str
    shot_id: str
    start_ms: int
    end_ms: int
    audio_path: Path
    sample_rate: int
    has_speech: bool
    language_hint: str | None

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("AudioSegment end_ms must be greater than start_ms.")
        if self.sample_rate != 16000:
            raise ValueError("AudioSegment sample_rate must be 16000.")
        if not self.audio_path.is_absolute():
            raise ValueError("AudioSegment audio_path must be absolute.")
        if not self.audio_path.is_file():
            raise FileNotFoundError(
                f"AudioSegment audio_path does not exist: {self.audio_path}"
            )
