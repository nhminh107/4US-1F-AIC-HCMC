"""Shared contracts for online visual question answering."""

from __future__ import annotations

from dataclasses import dataclass


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class VQAEvidence:
    """One image and its traceable retrieval identity supplied to a VLM."""

    image_path: str
    video_id: str
    timestamp_ms: int
    frame_id: str | None = None
    shot_id: str | None = None
    caption: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", _non_empty(self.image_path, "image_path"))
        object.__setattr__(self, "video_id", _non_empty(self.video_id, "video_id"))
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")


@dataclass(frozen=True, slots=True)
class VQARequest:
    """A grounded question over already-retrieved visual evidence."""

    question: str
    evidence: tuple[VQAEvidence, ...]
    max_evidence: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "question", _non_empty(self.question, "question"))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.evidence:
            raise ValueError("evidence must contain at least one image.")
        if self.max_evidence <= 0:
            raise ValueError("max_evidence must be greater than 0.")


@dataclass(frozen=True, slots=True)
class VQAResponse:
    """A VLM answer with model, prompt, and evidence traceability."""

    answer: str
    evidence: tuple[VQAEvidence, ...]
    model_name: str
    model_version: str
    prompt_version: str

    def __post_init__(self) -> None:
        for name in ("answer", "model_name", "model_version", "prompt_version"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        object.__setattr__(self, "evidence", tuple(self.evidence))
