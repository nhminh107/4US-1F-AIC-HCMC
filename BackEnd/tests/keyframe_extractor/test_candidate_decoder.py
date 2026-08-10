"""Tests for in-memory candidate decoding."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import DecodedFrameBatch
from BackEnd.app.keyframe_extractor.candidate_decoder import (
    CandidateDecodeError,
    PyAVCandidateFrameDecoder,
)


class FakeVideoDecoder:
    def __init__(self, batch: DecodedFrameBatch) -> None:
        self.batch = batch
        self.calls = []

    def decode_nearest_frames(self, video_asset, timestamps_ms):
        self.calls.append((video_asset, tuple(timestamps_ms)))
        return self.batch


class CandidateDecoderTests(unittest.TestCase):
    def test_default_decoder_disables_full_scan_fallback(self) -> None:
        decoder = PyAVCandidateFrameDecoder()

        self.assertFalse(decoder.decoder.allow_full_scan_fallback)

    def test_decodes_frame_indices_to_timestamp_requests(self) -> None:
        batch = DecodedFrameBatch(
            video_id="V001",
            images=(np.zeros((2, 2, 3), dtype=np.uint8),),
            requested_timestamps_ms=(1000,),
            actual_timestamps_ms=(1000,),
            decode_statuses=("success",),
        )
        fake = FakeVideoDecoder(batch)
        decoder = PyAVCandidateFrameDecoder(fake)

        frames = decoder.decode(video_id="V001", video_path=Path("video.mp4"), frame_indices=[25], fps=25.0)

        self.assertEqual(list(frames), [25])
        self.assertEqual(fake.calls[0][1], (1000,))

    def test_raises_when_any_candidate_fails(self) -> None:
        batch = DecodedFrameBatch(
            video_id="V001",
            images=(None,),
            requested_timestamps_ms=(1000,),
            actual_timestamps_ms=(None,),
            decode_statuses=("decode_failed",),
        )
        decoder = PyAVCandidateFrameDecoder(FakeVideoDecoder(batch))

        with self.assertRaises(CandidateDecodeError):
            decoder.decode(video_id="V001", video_path=Path("video.mp4"), frame_indices=[25], fps=25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
