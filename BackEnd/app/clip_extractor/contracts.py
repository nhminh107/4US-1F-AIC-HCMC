"""Input and output contracts used by :mod:`clip_extractor`.

The common pipeline passes records between modules.  These small dataclasses
validate the fields needed by this module while still allowing callers to pass
plain dictionaries or ORM-like objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .exceptions import InvalidShotError


RecordLike = Union[Mapping[str, Any], object]


def _read_field(record: RecordLike, field_name: str, *, required: bool = True) -> Any:
    if isinstance(record, Mapping):
        value = record.get(field_name)
    else:
        value = getattr(record, field_name, None)

    if required and value is None:
        raise InvalidShotError("Shot is missing required field: %s" % field_name)
    return value


def _as_non_empty_id(value: Any, field_name: str, *, max_length: int = 15) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise InvalidShotError("%s must not be empty" % field_name)
    if len(text) > max_length:
        raise InvalidShotError(
            "%s must contain at most %d characters" % (field_name, max_length)
        )
    return text


def _as_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise InvalidShotError("%s must be an integer" % field_name)

    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidShotError("%s must be an integer" % field_name) from exc

    if converted != value and not (
        isinstance(value, str) and value.strip() == str(converted)
    ):
        raise InvalidShotError("%s must not contain a fraction" % field_name)
    return converted


def _as_milliseconds(value: Any, field_name: str) -> int:
    return _as_integer(value, field_name)


@dataclass(frozen=True)
class ShotRecord:
    """The subset of the shared Shot contract required by ClipExtractor."""

    shot_id: str
    video_id: str
    start_ms: int
    end_ms: int
    start_frame_idx: int
    end_frame_idx: int
    video_path: Optional[Path] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @classmethod
    def from_contract(cls, record: Union["ShotRecord", RecordLike]) -> "ShotRecord":
        if isinstance(record, cls):
            return record

        shot_id = _as_non_empty_id(_read_field(record, "shot_id"), "shot_id")
        video_id = _as_non_empty_id(_read_field(record, "video_id"), "video_id")
        start_ms = _as_milliseconds(_read_field(record, "start_ms"), "start_ms")
        end_ms = _as_milliseconds(_read_field(record, "end_ms"), "end_ms")
        start_frame_idx = _as_integer(
            _read_field(record, "start_frame_idx"), "start_frame_idx"
        )
        end_frame_idx = _as_integer(
            _read_field(record, "end_frame_idx"), "end_frame_idx"
        )
        raw_video_path = _read_field(record, "video_path", required=False)

        if start_ms < 0:
            raise InvalidShotError("start_ms must be greater than or equal to 0")
        if end_ms <= start_ms:
            raise InvalidShotError("end_ms must be greater than start_ms")
        if start_frame_idx < 0:
            raise InvalidShotError(
                "start_frame_idx must be greater than or equal to 0"
            )
        if end_frame_idx <= start_frame_idx:
            raise InvalidShotError(
                "end_frame_idx must be greater than start_frame_idx"
            )

        video_path = None
        if raw_video_path is not None and str(raw_video_path).strip():
            video_path = Path(str(raw_video_path)).expanduser()

        return cls(
            shot_id=shot_id,
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
            video_path=video_path,
        )


@dataclass(frozen=True)
class ClipRecord:
    """A Clip produced from one long Shot.

    Time and frame positions are absolute in the original video.  Both use
    half-open intervals: ``[start, end)``.
    """

    clip_id: str
    shot_id: str
    video_id: str
    start_ms: int
    end_ms: int
    start_frame_idx: int
    end_frame_idx: int
    sampling_fps: float
    clip_index: int
    clip_path: Optional[Path] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_contract(self) -> Dict[str, Any]:
        """Return the DB/pipeline fields and omit runtime-only information."""

        return {
            "clip_id": self.clip_id,
            "shot_id": self.shot_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_frame_idx": self.start_frame_idx,
            "end_frame_idx": self.end_frame_idx,
            "sampling_fps": self.sampling_fps,
            "clip_path": str(self.clip_path) if self.clip_path else None,
        }

    def to_dict(self, *, include_runtime_fields: bool = True) -> Dict[str, Any]:
        """Serialize the result for logs, demos, or API responses."""

        data = self.to_contract()
        if include_runtime_fields:
            data.update(
                {
                    "video_id": self.video_id,
                    "duration_ms": self.duration_ms,
                    "clip_index": self.clip_index,
                }
            )
        return data
