"""Configuration for additional keyframe extraction strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from BackEnd.app.keyframe_extractor.sampling import (
    DEFAULT_MAX_ADDITIONAL_PER_SHOT,
    DEFAULT_MIN_FRAME_GAP,
)

KeyframeSelectionStrategy = Literal["time", "hybrid_clip", "hybrid_clip_strict"]


@dataclass(frozen=True, slots=True)
class HybridKeyframeConfig:
    """Runtime limits and thresholds for CLIP-based keyframe selection."""

    sample_fps: float = 1.0
    max_candidate_frames_per_shot: int = 64
    max_additional_per_shot: int = DEFAULT_MAX_ADDITIONAL_PER_SHOT
    min_frame_gap: int = DEFAULT_MIN_FRAME_GAP
    transition_margin_frames: int = 2
    hsv_similarity_threshold: float = 0.8
    clip_similarity_threshold: float = 0.95
    low_information_min_nonzero_bins: int = 10
    fallback_to_time_sampling: bool = True

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive.")
        if self.max_candidate_frames_per_shot <= 0:
            raise ValueError("max_candidate_frames_per_shot must be positive.")
        if self.max_additional_per_shot <= 0:
            raise ValueError("max_additional_per_shot must be positive.")
        if self.min_frame_gap < 0:
            raise ValueError("min_frame_gap must be non-negative.")
        if self.transition_margin_frames < 0:
            raise ValueError("transition_margin_frames must be non-negative.")
        if not 0 <= self.hsv_similarity_threshold <= 1:
            raise ValueError("hsv_similarity_threshold must be in [0, 1].")
        if not 0 <= self.clip_similarity_threshold <= 1:
            raise ValueError("clip_similarity_threshold must be in [0, 1].")
        if self.low_information_min_nonzero_bins < 0:
            raise ValueError("low_information_min_nonzero_bins must be non-negative.")


def normalize_strategy(strategy: KeyframeSelectionStrategy | str) -> KeyframeSelectionStrategy:
    """Validate a keyframe selection strategy string."""

    if strategy not in ("time", "hybrid_clip", "hybrid_clip_strict"):
        raise ValueError(f"Unknown keyframe selection strategy: {strategy!r}")
    return strategy  # type: ignore[return-value]
