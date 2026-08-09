from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BackEnd.app.audio_pre import extractor
from BackEnd.app.audio_pre import utils, vad
from BackEnd.app.audio_pre.exporter import save_audio_segments_json
from BackEnd.app.audio_pre.normalizer import normalize_audio
from BackEnd.app.audio_pre.run_preprocessing import (
    DEFAULT_OUTPUT_DIR,
    preprocess_full_video,
    preprocess_video,
    segment_shot,
)
from BackEnd.app.audio_pre.schemas import AudioSegment
from BackEnd.app.contracts.pipeline import ShotMetadata, VideoMetadata


class AudioPreprocessingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_wav(
        self,
        path: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        samples: np.ndarray | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if samples is None:
            samples = np.zeros(sample_rate // 10 * channels, dtype=np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.astype("<i2").tobytes())
        return path

    def test_validate_shots_rejects_invalid_boundaries_without_clamping(self) -> None:
        valid = ShotMetadata("shot-1", "video-1", 0, 0, 1000)
        utils.validate_shots([valid], 1000.0)

        invalid_cases = [
            ShotMetadata("negative", "video-1", 0, -1, 100),
            ShotMetadata("empty", "video-1", 0, 100, 100),
            ShotMetadata("starts-at-duration", "video-1", 0, 1000, 1100),
            ShotMetadata("exceeds-duration", "video-1", 0, 500, 1001),
        ]
        for shot in invalid_cases:
            with self.subTest(shot=shot.shot_id), self.assertRaises(ValueError):
                utils.validate_shots([shot], 1000.0)

    def test_has_audio_stream_uses_ffprobe_stream_result(self) -> None:
        with patch(
            "BackEnd.app.audio_pre.extractor.utils.run_command",
            return_value=subprocess.CompletedProcess([], 0, stdout=b'{"streams":[{"index":0}]}', stderr=b""),
        ):
            self.assertTrue(extractor.has_audio_stream(self.root / "video.mp4"))

        with patch(
            "BackEnd.app.audio_pre.extractor.utils.run_command",
            return_value=subprocess.CompletedProcess([], 0, stdout=b'{"streams":[]}', stderr=b""),
        ):
            self.assertFalse(extractor.has_audio_stream(self.root / "video.mp4"))

    def test_get_or_extract_raw_audio_reuses_valid_existing_wav(self) -> None:
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        raw_path = self._write_wav(self.root / "video-1" / "video-1_raw.wav")

        with patch("BackEnd.app.audio_pre.extractor.extract_full_audio") as extract:
            self.assertEqual(
                extractor.get_or_extract_raw_audio(video, self.root),
                raw_path.resolve(),
            )

        extract.assert_not_called()

    def test_get_or_extract_raw_audio_discards_invalid_raw_before_reextracting(self) -> None:
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        raw_path = self.root / "video-1" / "video-1_raw.wav"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(b"broken")

        replacement = self._write_wav(self.root / "video-1" / "replacement.wav")
        with patch(
            "BackEnd.app.audio_pre.extractor.extract_full_audio",
            return_value=replacement,
        ) as extract:
            self.assertEqual(extractor.get_or_extract_raw_audio(video, self.root), replacement)

        extract.assert_called_once_with(video, self.root)
        self.assertFalse(raw_path.exists())

    def test_segment_shot_failure_removes_partial_artifact(self) -> None:
        raw_path = self._write_wav(self.root / "raw.wav")
        output_path = self.root / "partial.wav"
        output_path.write_bytes(b"partial")

        with patch(
            "BackEnd.app.audio_pre.run_preprocessing.utils.run_command",
            return_value=subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"failed"),
        ):
            with self.assertRaises(utils.AudioSegmentationError):
                segment_shot(raw_path, 0, 100, output_path)

        self.assertFalse(output_path.exists())

    def test_audio_segment_schema_and_export_are_stable(self) -> None:
        audio_path = self._write_wav(self.root / "video-1" / "video-1_shot0002.wav")
        segment = AudioSegment(
            segment_id="video-1_shot0002",
            video_id="video-1",
            shot_id="shot-2",
            start_ms=10,
            end_ms=40,
            audio_path=audio_path.resolve(),
            sample_rate=16000,
            has_speech=False,
            language_hint="vi",
        )

        output_path = save_audio_segments_json([segment], self.root, "video-1")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["video_id"], "video-1")
        self.assertEqual(payload["segments"][0]["segment_id"], "video-1_shot0002")
        self.assertEqual(payload["segments"][0]["audio_path"], str(audio_path.resolve()))
        self.assertEqual(payload["segments"][0]["language_hint"], "vi")

    def test_vad_uses_complete_30_ms_frames_only(self) -> None:
        samples = np.zeros(640, dtype=np.int16)
        audio_path = self._write_wav(self.root / "vad.wav", samples=samples)

        class FakeVAD:
            def __init__(self) -> None:
                self.calls = 0

            def is_speech(self, frame: bytes, sample_rate: int) -> bool:
                self.calls += 1
                self.last_frame_length = len(frame)
                self.last_sample_rate = sample_rate
                return False

        fake_vad = FakeVAD()
        with patch("BackEnd.app.audio_pre.vad._create_vad", return_value=fake_vad):
            self.assertFalse(vad.detect_speech(audio_path))

        self.assertEqual(fake_vad.calls, 1)
        self.assertEqual(fake_vad.last_frame_length, 960)
        self.assertEqual(fake_vad.last_sample_rate, 16000)

    def test_preprocess_video_returns_empty_for_video_without_audio(self) -> None:
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        with patch("BackEnd.app.audio_pre.run_preprocessing.extractor.has_audio_stream", return_value=False):
            self.assertEqual(preprocess_video(video, [], self.root), [])

    def test_preprocess_full_video_wraps_complete_audio_as_one_shot(self) -> None:
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        raw_path = self._write_wav(self.root / "video-1_raw.wav")
        expected_segments: list[AudioSegment] = []

        with (
            patch(
                "BackEnd.app.audio_pre.run_preprocessing.extractor.has_audio_stream",
                return_value=True,
            ),
            patch(
                "BackEnd.app.audio_pre.run_preprocessing.extractor.get_or_extract_raw_audio",
                return_value=raw_path,
            ),
            patch(
                "BackEnd.app.audio_pre.run_preprocessing.extractor.get_duration_ms",
                return_value=2500.75,
            ),
            patch(
                "BackEnd.app.audio_pre.run_preprocessing.preprocess_video",
                return_value=expected_segments,
            ) as preprocess,
        ):
            segments = preprocess_full_video(video)

        self.assertIs(segments, expected_segments)
        preprocess.assert_called_once()
        wrapped_video, wrapped_shots, wrapped_output_dir = preprocess.call_args.args
        self.assertIs(wrapped_video, video)
        self.assertEqual(wrapped_output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(
            wrapped_shots,
            [
                ShotMetadata(
                    shot_id="video-1",
                    video_id="video-1",
                    shot_index=0,
                    start_ms=0,
                    end_ms=2500,
                )
            ],
        )
        self.assertIsNone(preprocess.call_args.kwargs["language_hint"])
        self.assertFalse(raw_path.exists())

    def test_preprocess_video_skips_failed_shot_and_deletes_raw(self) -> None:
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        raw_path = self._write_wav(self.root / "video-1" / "video-1_raw.wav")
        good_audio = self._write_wav(self.root / "video-1" / "video-1_shot0000.wav")
        good_segment = AudioSegment(
            segment_id="video-1_shot0000",
            video_id="video-1",
            shot_id="shot-0",
            start_ms=0,
            end_ms=500,
            audio_path=good_audio.resolve(),
            sample_rate=16000,
            has_speech=True,
            language_hint=None,
        )
        shots = [
            ShotMetadata("shot-0", "video-1", 0, 0, 500),
            ShotMetadata("shot-1", "video-1", 1, 500, 1000),
        ]

        with (
            patch("BackEnd.app.audio_pre.run_preprocessing.extractor.has_audio_stream", return_value=True),
            patch("BackEnd.app.audio_pre.run_preprocessing.extractor.get_or_extract_raw_audio", return_value=raw_path),
            patch("BackEnd.app.audio_pre.run_preprocessing.extractor.get_duration_ms", return_value=1000.0),
            patch(
                "BackEnd.app.audio_pre.run_preprocessing.preprocess_shot",
                side_effect=[good_segment, RuntimeError("boom")],
            ),
        ):
            segments = preprocess_video(video, shots, self.root)

        self.assertEqual(segments, [good_segment])
        self.assertFalse(raw_path.exists())
        payload = json.loads((self.root / "video-1" / "audio_segments.json").read_text())
        self.assertEqual(len(payload["segments"]), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for normalization")
    def test_normalizer_outputs_16khz_mono_pcm16_peak_normalized_wav(self) -> None:
        sample_rate = 44100
        t = np.linspace(0, 0.1, sample_rate // 10, endpoint=False)
        left = (np.sin(2 * np.pi * 440 * t) * 1000).astype(np.int16)
        right = (np.sin(2 * np.pi * 440 * t) * 2000).astype(np.int16)
        stereo = np.column_stack([left, right]).ravel()
        input_path = self._write_wav(
            self.root / "input.wav",
            sample_rate=sample_rate,
            channels=2,
            samples=stereo,
        )
        output_path = normalize_audio(input_path, self.root / "out.wav")

        with wave.open(str(output_path), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")

        self.assertGreater(data.size, 0)
        self.assertEqual(int(np.max(np.abs(data))), 32767)


if __name__ == "__main__":
    unittest.main(verbosity=2)
