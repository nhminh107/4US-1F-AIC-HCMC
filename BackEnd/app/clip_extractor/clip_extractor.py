"""Split long Shots into smaller Clip records and optionally create MP4 files."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from BackEnd.CONFIG import ClipExtractorConfig

from .contracts import ClipRecord, RecordLike, ShotRecord
from .exceptions import (
    ClipMaterializationError,
    FFmpegNotAvailableError,
    InvalidShotError,
    SourceVideoError,
)


ClipIdFactory = Callable[[ShotRecord, int], str]
CommandRunner = Callable[..., subprocess.CompletedProcess]


_CANONICAL_SHOT_ID = re.compile(r"^(L\d+)_V(\d+)_S(\d+)$", re.IGNORECASE)


def _default_clip_id(shot: ShotRecord, one_based_index: int) -> str:
    """Create a deterministic ID that fits ``clipwindow.clip_id varchar(15)``."""

    match = _CANONICAL_SHOT_ID.fullmatch(shot.shot_id)
    if match and one_based_index <= 99:
        candidate = "%sV%sS%sC%02d" % (
            match.group(1).upper(),
            match.group(2),
            match.group(3),
            one_based_index,
        )
        if len(candidate) <= 15:
            return candidate

    # Stable fallback for non-standard Shot IDs.  The leading C identifies a
    # Clip while 14 hexadecimal characters keep the value within varchar(15).
    source = "%s:%d" % (shot.shot_id, one_based_index)
    return "C" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:14].upper()


class ClipExtractor:
    """Reusable Clip Extractor module for the common offline pipeline.

    Standard behavior from ``pipeline_offline.md``:

    * a Shot of 10 seconds or less becomes one full-Shot Clip;
    * a longer Shot becomes fixed-duration windows with contextual overlap;
    * output records contain every column required by ``clipwindow``;
    * records are produced without saving video files unless explicitly enabled.

    Keeping file creation optional avoids duplicating large video data.  Downstream
    modules can read a Clip from the source video using ``video_id``, ``start_ms``
    and ``end_ms``.  When a real MP4 is needed for a demo or external model,
    ``materialize_files=True`` uses FFmpeg to create it accurately.
    """

    def __init__(
        self,
        config: Optional[ClipExtractorConfig] = None,
        *,
        clip_id_factory: Optional[ClipIdFactory] = None,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self.config = config or ClipExtractorConfig()
        self._clip_id_factory = clip_id_factory or _default_clip_id
        self._command_runner = command_runner

    def run(
        self,
        shot: Union[ShotRecord, RecordLike],
        *,
        video_path: Optional[Union[str, Path]] = None,
        output_root: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of Clip metadata objects ready for the pipeline/DB."""

        shot_record = ShotRecord.from_contract(shot)
        clips = self.prepare_clips(shot_record)

        if not self.config.materialize_files or not clips:
            return [clip.to_contract() for clip in clips]

        resolved_source = self._resolve_video_path(shot_record, video_path)
        resolved_output_root = Path(output_root or self.config.output_root)
        self._validate_media_tools()

        if self.config.validate_source_duration:
            source_duration_ms = self._probe_duration_ms(resolved_source)
            if shot_record.end_ms > source_duration_ms + 100:
                raise SourceVideoError(
                    "Shot end_ms (%d) is after source video duration (%d ms)"
                    % (shot_record.end_ms, source_duration_ms)
                )

        materialized = []
        for clip in clips:
            output_path = (
                resolved_output_root
                / self._safe_path_component(shot_record.video_id)
                / (self._safe_path_component(clip.clip_id) + ".mp4")
            )
            if len(str(output_path)) > 200:
                raise ClipMaterializationError(
                    "clip_path exceeds clipwindow.clip_path varchar(200): %s"
                    % output_path
                )
            if self.config.overwrite or not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._materialize_one(resolved_source, output_path, clip)

            materialized.append(
                ClipRecord(
                    clip_id=clip.clip_id,
                    shot_id=clip.shot_id,
                    video_id=clip.video_id,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                    start_frame_idx=clip.start_frame_idx,
                    end_frame_idx=clip.end_frame_idx,
                    sampling_fps=clip.sampling_fps,
                    clip_index=clip.clip_index,
                    clip_path=output_path,
                )
            )

        return [clip.to_contract() for clip in materialized]

    def prepare_clips(
        self, shot: Union[ShotRecord, RecordLike]
    ) -> List[ClipRecord]:
        """Create Clip metadata only; this method performs no video I/O."""

        shot_record = ShotRecord.from_contract(shot)
        intervals = plan_clip_windows(
            shot_record.start_ms,
            shot_record.end_ms,
            split_threshold_ms=self.config.split_threshold_ms,
            window_ms=self.config.max_clip_duration_ms,
            stride_ms=self.config.stride_ms,
            min_new_window_gap_ms=self.config.min_new_window_gap_ms,
        )
        sampling_fps = self._sampling_fps(shot_record)
        clips = []
        clip_ids = set()

        for zero_based_index, (start_ms, end_ms, _) in enumerate(intervals):
            one_based_index = zero_based_index + 1
            clip_id = str(self._clip_id_factory(shot_record, one_based_index)).strip()
            if not clip_id:
                raise InvalidShotError("clip_id_factory returned an empty clip_id")
            if len(clip_id) > 15:
                raise InvalidShotError(
                    "clip_id must contain at most 15 characters for clipwindow"
                )
            if clip_id in clip_ids:
                raise InvalidShotError("clip_id_factory returned a duplicate clip_id")
            clip_ids.add(clip_id)

            start_frame_idx = self._frame_boundary(shot_record, start_ms)
            end_frame_idx = self._frame_boundary(shot_record, end_ms)

            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    shot_id=shot_record.shot_id,
                    video_id=shot_record.video_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    start_frame_idx=start_frame_idx,
                    end_frame_idx=end_frame_idx,
                    sampling_fps=sampling_fps,
                    clip_index=one_based_index,
                )
            )

        self._assert_valid_windows(shot_record, clips)
        return clips

    def _sampling_fps(self, shot: ShotRecord) -> float:
        if self.config.sampling_fps is not None:
            return float(self.config.sampling_fps)
        return (
            (shot.end_frame_idx - shot.start_frame_idx)
            * 1000.0
            / shot.duration_ms
        )

    @staticmethod
    def _frame_boundary(shot: ShotRecord, boundary_ms: int) -> int:
        elapsed_ms = boundary_ms - shot.start_ms
        frame_span = shot.end_frame_idx - shot.start_frame_idx
        return shot.start_frame_idx + round(
            frame_span * elapsed_ms / shot.duration_ms
        )

    def _resolve_video_path(
        self,
        shot: ShotRecord,
        explicit_video_path: Optional[Union[str, Path]],
    ) -> Path:
        candidate = (
            Path(explicit_video_path) if explicit_video_path else shot.video_path
        )
        if candidate is None:
            raise SourceVideoError(
                "video_path is required when materialize_files=True"
            )
        candidate = candidate.expanduser().resolve()
        if not candidate.is_file():
            raise SourceVideoError("Source video does not exist: %s" % candidate)
        return candidate

    def _validate_media_tools(self) -> None:
        missing = [
            binary
            for binary in (self.config.ffmpeg_bin, self.config.ffprobe_bin)
            if shutil.which(binary) is None
        ]
        if missing:
            raise FFmpegNotAvailableError(
                "Missing required media tool(s): %s" % ", ".join(missing)
            )

    def _probe_duration_ms(self, video_path: Path) -> int:
        command = [
            self.config.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            completed = self._command_runner(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            duration_seconds = float(payload["format"]["duration"])
        except (
            subprocess.CalledProcessError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise SourceVideoError(
                "Cannot read source video duration: %s" % video_path
            ) from exc
        return int(round(duration_seconds * 1000))

    def _materialize_one(
        self, source_path: Path, output_path: Path, clip: ClipRecord
    ) -> None:
        start_seconds = self._seconds_arg(clip.start_ms)
        duration_seconds = self._seconds_arg(clip.duration_ms)
        command = [
            self.config.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self.config.overwrite else "-n",
            "-i",
            str(source_path),
            "-ss",
            start_seconds,
            "-t",
            duration_seconds,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        if self.config.sampling_fps is not None:
            command.extend(["-vf", "fps=%.6g" % self.config.sampling_fps])
        command.extend(
            [
                "-c:v",
                self.config.video_codec,
                "-preset",
                self.config.preset,
                "-crf",
                str(self.config.crf),
                "-c:a",
                self.config.audio_codec,
                "-movflags",
                "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                str(output_path),
            ]
        )
        try:
            self._command_runner(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output_path.unlink(missing_ok=True)
            stderr = (exc.stderr or "").strip()
            detail = stderr[-1000:] if stderr else "unknown FFmpeg error"
            raise ClipMaterializationError(
                "Cannot create clip %s: %s" % (clip.clip_id, detail)
            ) from exc

        if not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise ClipMaterializationError(
                "FFmpeg did not create a valid file for clip %s" % clip.clip_id
            )

    @staticmethod
    def _seconds_arg(milliseconds: int) -> str:
        return "%.3f" % (milliseconds / 1000.0)

    @staticmethod
    def _safe_path_component(value: str) -> str:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in value
        ).strip("._")
        return safe or "unnamed"

    def _assert_valid_windows(
        self, shot: ShotRecord, clips: Sequence[ClipRecord]
    ) -> None:
        if not clips:
            raise AssertionError("Every Shot must produce at least one Clip")
        if clips[-1].end_ms != shot.end_ms:
            raise AssertionError("The final Clip must reach the end of the Shot")

        if clips[0].start_ms != shot.start_ms:
            raise AssertionError("The first Clip must start with its parent Shot")

        previous_start = shot.start_ms - 1
        previous_end: Optional[int] = None
        for clip in clips:
            if clip.start_ms < shot.start_ms or clip.end_ms > shot.end_ms:
                raise AssertionError("A Clip is outside its parent Shot")
            if clip.start_ms <= previous_start:
                raise AssertionError("Clip windows are not strictly ordered")
            if previous_end is not None and clip.start_ms > previous_end:
                raise AssertionError("Clip windows contain an uncovered gap")
            if clip.duration_ms <= 0:
                raise AssertionError("Every Clip must have a positive duration")
            if clip.duration_ms > self.config.max_clip_duration_ms:
                raise AssertionError("A Clip exceeds max_clip_duration_ms")
            if clip.end_frame_idx <= clip.start_frame_idx:
                raise AssertionError("Every Clip must contain at least one frame")
            previous_start = clip.start_ms
            previous_end = clip.end_ms


def plan_clip_windows(
    start_ms: int,
    end_ms: int,
    *,
    split_threshold_ms: int = 10_000,
    window_ms: int = 10_000,
    stride_ms: int = 8_000,
    min_new_window_gap_ms: int = 2_000,
) -> List[tuple[int, int, str]]:
    """Plan logical Clip intervals without decoding or materializing video.

    Short Shots produce one ``full_shot`` interval. Longer Shots use overlapping
    fixed windows and align the final window to the Shot end. If the tail would
    add an almost duplicate window, the preceding window is tail-aligned.
    """

    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("A Clip window range must satisfy 0 <= start_ms < end_ms")
    if split_threshold_ms <= 0 or window_ms <= 0:
        raise ValueError("split_threshold_ms and window_ms must be greater than 0")
    if split_threshold_ms > window_ms:
        raise ValueError("split_threshold_ms must be less than or equal to window_ms")
    if stride_ms <= 0 or stride_ms > window_ms:
        raise ValueError("stride_ms must satisfy 0 < stride_ms <= window_ms")
    if min_new_window_gap_ms <= 0 or min_new_window_gap_ms > stride_ms:
        raise ValueError(
            "min_new_window_gap_ms must satisfy "
            "0 < min_new_window_gap_ms <= stride_ms"
        )

    if end_ms - start_ms <= split_threshold_ms:
        return [(start_ms, end_ms, "full_shot")]

    intervals: List[tuple[int, int, str]] = []
    current_start = start_ms
    while current_start + window_ms < end_ms:
        intervals.append(
            (current_start, current_start + window_ms, "fixed_window")
        )
        current_start += stride_ms

    tail_start = end_ms - window_ms
    if (
        len(intervals) <= 1
        or tail_start - intervals[-1][0] >= min_new_window_gap_ms
    ):
        intervals.append((tail_start, end_ms, "fixed_window"))
    else:
        intervals[-1] = (tail_start, end_ms, "fixed_window")

    return intervals
