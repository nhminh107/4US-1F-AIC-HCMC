"""Tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    sampling_fps: float = 2.0
    track_activation_threshold: float = 0.25
    high_confidence_threshold: float = 0.35
    minimum_iou_threshold: float = 0.20
    lost_track_buffer: int = 30

    def __post_init__(self) -> None:
        if self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be positive.")
