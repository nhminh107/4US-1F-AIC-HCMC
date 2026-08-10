"""ChunkFormer-based ASR that returns shared transcript data contracts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from chunkformer import ChunkFormerModel

from BackEnd.CONFIG import (
    ASR_AUDIO_OUTPUT_DIR as DEFAULT_AUDIO_OUTPUT_DIR,
    ASR_CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
    ASR_LANGUAGE as DEFAULT_LANGUAGE,
    ASR_LEFT_CONTEXT_SIZE as DEFAULT_LEFT_CONTEXT_SIZE,
    ASR_MODEL_ID as DEFAULT_MODEL_ID,
    ASR_RIGHT_CONTEXT_SIZE as DEFAULT_RIGHT_CONTEXT_SIZE,
    ASR_TOTAL_BATCH_DURATION_SECONDS as DEFAULT_TOTAL_BATCH_DURATION,
)
from BackEnd.app.audio_pre import preprocess_full_video
from BackEnd.app.audio_pre.schemas import AudioSegment
from BackEnd.app.contracts.pipeline import TranscriptSegmentResult, VideoMetadata

logger = logging.getLogger(__name__)

class ASR_Model:
    """Transcribe normalized full-video audio with a reusable ChunkFormer model."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        audio_output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
        language: str = DEFAULT_LANGUAGE,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_output_dir = Path(audio_output_dir)
        self.language = language
        self.model = ChunkFormerModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded ASR model on device: %s", self.device)

    @staticmethod
    def _timestamp_to_ms(timestamp: str) -> int:
        """Convert ChunkFormer's ``HH:MM:SS:mmm`` timestamp to milliseconds."""

        try:
            hours, minutes, seconds, milliseconds = map(int, timestamp.split(":"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid ChunkFormer timestamp '{timestamp}'; expected HH:MM:SS:mmm."
            ) from exc

        if min(hours, minutes, seconds, milliseconds) < 0:
            raise ValueError(f"Timestamp must not contain negative values: '{timestamp}'.")
        return (
            hours * 3_600_000
            + minutes * 60_000
            + seconds * 1_000
            + milliseconds
        )

    @staticmethod
    def _connect_short_transcripts(
        transcription: list[dict[str, Any]],
        *,
        minimum_word_count: int = 8,
    ) -> list[dict[str, str]]:
        """Merge adjacent short timestamped sentences without dropping trailing text."""

        connected: list[dict[str, str]] = []
        pending: dict[str, str] | None = None

        for item in transcription:
            text = str(item.get("decode", "")).strip()
            start = item.get("start")
            end = item.get("end")
            if not text or not isinstance(start, str) or not isinstance(end, str):
                continue

            word_count = len(text.split())
            if pending is None and word_count >= minimum_word_count:
                connected.append({"decode": text, "start": start, "end": end})
                continue

            if pending is None:
                pending = {"decode": text, "start": start, "end": end}
            else:
                pending["decode"] = f"{pending['decode']} {text}"
                pending["end"] = end

            if len(pending["decode"].split()) >= minimum_word_count:
                connected.append(pending)
                pending = None

        if pending is not None:
            connected.append(pending)
        return connected

    def _prepare_audio(self, video: VideoMetadata) -> list[AudioSegment]:
        segments = preprocess_full_video(video, output_dir=self.audio_output_dir)
        return [segment for segment in segments if segment.has_speech]

    def _timestamped_results(
        self,
        video: VideoMetadata,
        audio_segment: AudioSegment,
        transcription: list[dict[str, Any]],
    ) -> list[TranscriptSegmentResult]:
        results: list[TranscriptSegmentResult] = []
        for index, item in enumerate(self._connect_short_transcripts(transcription)):
            start_ms = audio_segment.start_ms + self._timestamp_to_ms(item["start"])
            end_ms = audio_segment.start_ms + self._timestamp_to_ms(item["end"])
            start_ms = max(start_ms, audio_segment.start_ms)
            end_ms = min(end_ms, audio_segment.end_ms)
            if end_ms <= start_ms:
                logger.warning(
                    "Skipping invalid ASR timestamp range: video_id=%s start_ms=%s end_ms=%s",
                    video.video_id,
                    start_ms,
                    end_ms,
                )
                continue

            results.append(
                TranscriptSegmentResult(
                    segment_id=f"{video.video_id}_{index:04d}",
                    video_id=video.video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=item["decode"],
                    language=self.language,
                )
            )
        return results

    def transcript_segment(self, video: VideoMetadata) -> list[TranscriptSegmentResult]:
        """Transcribe one video with sentence-level timestamps."""

        audio_segments = self._prepare_audio(video)
        results: list[TranscriptSegmentResult] = []
        for audio_segment in audio_segments:
            transcription = self.model.endless_decode(
                audio_path=str(audio_segment.audio_path),
                chunk_size=DEFAULT_CHUNK_SIZE,
                left_context_size=DEFAULT_LEFT_CONTEXT_SIZE,
                right_context_size=DEFAULT_RIGHT_CONTEXT_SIZE,
                total_batch_duration=DEFAULT_TOTAL_BATCH_DURATION,
                return_timestamps=True,
            )
            if not isinstance(transcription, list):
                raise TypeError("ChunkFormer endless_decode must return a list of timestamped records.")
            results.extend(self._timestamped_results(video, audio_segment, transcription))
        return results

    def batch_transcript(
        self,
        videos: list[VideoMetadata],
    ) -> list[TranscriptSegmentResult]:
        """Batch-transcribe videos and return one full-duration result per non-empty audio.

        ChunkFormer's ``batch_decode`` returns text without sentence timestamps.
        Each result therefore keeps the exact full range of its source
        ``AudioSegment`` rather than fabricating sentence boundaries.
        """

        speech_segments: list[AudioSegment] = []
        for video in videos:
            speech_segments.extend(self._prepare_audio(video))

        if not speech_segments:
            return []

        decodes = self.model.batch_decode(
            audio_paths=[str(segment.audio_path) for segment in speech_segments],
            chunk_size=DEFAULT_CHUNK_SIZE,
            left_context_size=DEFAULT_LEFT_CONTEXT_SIZE,
            right_context_size=DEFAULT_RIGHT_CONTEXT_SIZE,
            total_batch_duration=DEFAULT_TOTAL_BATCH_DURATION,
        )
        if len(decodes) != len(speech_segments):
            raise RuntimeError(
                "ChunkFormer batch_decode returned a different number of results than inputs: "
                f"{len(decodes)} != {len(speech_segments)}."
            )

        results: list[TranscriptSegmentResult] = []
        per_video_index: dict[str, int] = {}
        for audio_segment, decode in zip(speech_segments, decodes, strict=True):
            text = str(decode).strip()
            if not text:
                continue

            result_index = per_video_index.get(audio_segment.video_id, 0)
            per_video_index[audio_segment.video_id] = result_index + 1
            results.append(
                TranscriptSegmentResult(
                    segment_id=f"{audio_segment.video_id}_{result_index:04d}",
                    video_id=audio_segment.video_id,
                    start_ms=audio_segment.start_ms,
                    end_ms=audio_segment.end_ms,
                    text=text,
                    language=self.language,
                )
            )
        return results
