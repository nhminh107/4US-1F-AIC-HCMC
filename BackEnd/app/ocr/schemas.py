"""Private typed values exchanged inside the OCR module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectedTextRegion:
    """One detector polygon expressed in original-image pixel coordinates."""

    polygon: np.ndarray
    confidence: float


@dataclass(frozen=True, slots=True)
class RecognizedText:
    """Raw output of the text recognition model."""

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PreparedTextRegion:
    """A detected region paired with its perspective-corrected crop."""

    frame_index: int
    polygon: np.ndarray
    detection_confidence: float
    cropped_image: np.ndarray

