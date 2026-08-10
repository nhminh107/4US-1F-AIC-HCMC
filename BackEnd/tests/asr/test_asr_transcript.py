from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from BackEnd.app.audio_pre.schemas import AudioSegment
from BackEnd.app.contracts.pipeline import VideoMetadata


class ASRTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _audio_segment(self, video_id: str, *, has_speech: bool = True) -> AudioSegment:
        audio_path = self.root / f"{video_id}.wav"
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        return AudioSegment(
            segment_id=f"{video_id}_shot0000",
            video_id=video_id,
            shot_id=video_id,
            start_ms=0,
            end_ms=1000,
            audio_path=audio_path.resolve(),
            sample_rate=16000,
            has_speech=has_speech,
            language_hint=None,
        )

    def _model(self) -> MagicMock:
        model = MagicMock()
        model.to.return_value = model
        return model

    @patch("BackEnd.app.ASR.asr_transcript.ChunkFormerModel.from_pretrained")
    def test_batch_transcript_uses_batch_decode_and_keeps_contract(
        self,
        load_model: MagicMock,
    ) -> None:
        from BackEnd.app.ASR.asr_transcript import ASR_Model

        model = self._model()
        model.batch_decode.return_value = ["xin chao"]
        load_model.return_value = model
        first = VideoMetadata("video-1", self.root / "video-1.mp4")
        second = VideoMetadata("video-2", self.root / "video-2.mp4")
        first_audio = self._audio_segment("video-1")

        with patch(
            "BackEnd.app.ASR.asr_transcript.preprocess_full_video",
            side_effect=[[first_audio], []],
        ):
            asr_model = ASR_Model(device=torch.device("cpu"))
            results = asr_model.batch_transcript([first, second])

        model.batch_decode.assert_called_once()
        self.assertEqual(
            model.batch_decode.call_args.kwargs["audio_paths"],
            [str(first_audio.audio_path)],
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.segment_id, "video-1_0000")
        self.assertEqual(result.video_id, "video-1")
        self.assertEqual((result.start_ms, result.end_ms), (0, 1000))
        self.assertEqual(result.text, "xin chao")
        self.assertEqual(result.language, "vi")

    @patch("BackEnd.app.ASR.asr_transcript.ChunkFormerModel.from_pretrained")
    def test_transcript_segment_merges_short_tail_and_returns_timestamps(
        self,
        load_model: MagicMock,
    ) -> None:
        from BackEnd.app.ASR.asr_transcript import ASR_Model

        model = self._model()
        model.endless_decode.return_value = [
            {"decode": "mot hai", "start": "00:00:00:000", "end": "00:00:00:200"},
            {
                "decode": "ba bon nam sau bay tam chin",
                "start": "00:00:00:200",
                "end": "00:00:00:900",
            },
        ]
        load_model.return_value = model
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        audio = self._audio_segment("video-1")

        with patch("BackEnd.app.ASR.asr_transcript.preprocess_full_video", return_value=[audio]):
            asr_model = ASR_Model(device=torch.device("cpu"))
            results = asr_model.transcript_segment(video)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "mot hai ba bon nam sau bay tam chin")
        self.assertEqual((results[0].start_ms, results[0].end_ms), (0, 900))

    @patch("BackEnd.app.ASR.asr_transcript.ChunkFormerModel.from_pretrained")
    def test_batch_transcript_skips_no_speech_audio(self, load_model: MagicMock) -> None:
        from BackEnd.app.ASR.asr_transcript import ASR_Model

        model = self._model()
        load_model.return_value = model
        video = VideoMetadata("video-1", self.root / "video-1.mp4")
        silent_audio = self._audio_segment("video-1", has_speech=False)

        with patch("BackEnd.app.ASR.asr_transcript.preprocess_full_video", return_value=[silent_audio]):
            asr_model = ASR_Model(device=torch.device("cpu"))
            self.assertEqual(asr_model.batch_transcript([video]), [])

        model.batch_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
