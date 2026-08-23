"""Tests for hybrid CLIP keyframe selection orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.config import HybridKeyframeConfig
from BackEnd.app.keyframe_extractor.hybrid_selector import (
    HybridKeyframeSelectionError,
    HybridKeyframeSelector,
    _evenly_limit_indices,
)


def textured(value: int) -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = [value, 255 - value, 40]
    image[:, 8:] = [255 - value, value, 220]
    return image


class FakeDecoder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.last_frame_indices = []

    def decode(self, *, video_id, video_path, frame_indices, fps):
        if self.fail:
            raise RuntimeError("decode failed")
        self.last_frame_indices = list(frame_indices)
        return {frame_idx: textured((frame_idx * 7) % 255) for frame_idx in frame_indices}


class FakeEmbeddingAdapter:
    def encode_images(self, images):
        vectors = []
        for index, _image in enumerate(images):
            vectors.append([float(index + 1), 1.0 if index % 2 else 0.0])
        return np.asarray(vectors, dtype=np.float32)


class HybridSelectorTests(unittest.TestCase):
    def test_selects_clustered_non_redundant_frames(self) -> None:
        config = HybridKeyframeConfig(
            sample_fps=1.0,
            max_candidate_frames_per_shot=8,
            max_additional_per_shot=3,
            low_information_min_nonzero_bins=2,
        )
        decoder = FakeDecoder()
        selector = HybridKeyframeSelector(
            config=config,
            decoder=decoder,
            embedding_adapter=FakeEmbeddingAdapter(),
        )
        shot = ShotMetadata("V001_S001", "V001", 0, 0, 5000, 0, 124)

        selected = selector.select(shot, video_path=Path("video.mp4"), fps=25.0, existing_frame_idxs=[])

        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(selected, sorted(selected))
        self.assertGreater(len(decoder.last_frame_indices), len(selected))
        self.assertEqual(selector.last_metrics["status"], "success")
        self.assertEqual(selector.last_metrics["candidate_count"], len(decoder.last_frame_indices))
        self.assertEqual(selector.last_metrics["selected_count"], len(selected))
        self.assertGreaterEqual(selector.last_metrics["decode_s"], 0.0)

    def test_wraps_decode_failure(self) -> None:
        selector = HybridKeyframeSelector(
            decoder=FakeDecoder(fail=True),
            embedding_adapter=FakeEmbeddingAdapter(),
        )
        shot = ShotMetadata("V001_S001", "V001", 0, 0, 5000, 0, 124)

        with self.assertRaises(HybridKeyframeSelectionError):
            selector.select(shot, video_path=Path("video.mp4"), fps=25.0, existing_frame_idxs=[])

    def test_bounds_reference_frame_work_without_losing_temporal_extent(self) -> None:
        limited = _evenly_limit_indices(list(range(100)), 16)

        self.assertEqual(len(limited), 16)
        self.assertEqual(limited[0], 0)
        self.assertEqual(limited[-1], 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
