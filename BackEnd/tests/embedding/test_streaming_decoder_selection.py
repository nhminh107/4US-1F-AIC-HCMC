from __future__ import annotations

import numpy as np

from BackEnd.app.embedding.clip.decoder import _select_nearest_frames


class FakeFrame:
    def __init__(self, value: int) -> None:
        self.value = value
        self.conversions = 0

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        self.conversions += 1
        return np.full((1, 1, 3), self.value, dtype=np.uint8)


def test_select_nearest_frames_only_converts_selected_candidates() -> None:
    frames = [(timestamp, FakeFrame(timestamp)) for timestamp in range(0, 1_001, 100)]

    selected, decoded_count = _select_nearest_frames(
        frames,
        [240, 760],
        start_ms=0,
        end_ms=1_000,
    )

    assert decoded_count == 11
    assert selected[240][0] == 200
    assert selected[760][0] == 800
    assert sum(frame.conversions for _, frame in frames) == 2


def test_select_nearest_frames_reuses_one_rgb_conversion() -> None:
    shared = FakeFrame(500)

    selected, _ = _select_nearest_frames(
        [(500, shared)],
        [450, 500, 550],
        start_ms=400,
        end_ms=600,
    )

    assert set(selected) == {450, 500, 550}
    assert shared.conversions == 1
