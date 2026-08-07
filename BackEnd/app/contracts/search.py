"""Typed contracts for Elasticsearch text indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TextSourceType = Literal["video_metadata", "ocr", "transcript", "caption"]

TEXT_SOURCE_TYPES: frozenset[str] = frozenset(
    {"video_metadata", "ocr", "transcript", "caption"}
)
OCR_REGIONS: frozenset[str] = frozenset(
    {"top", "header", "bottom", "footer", "left", "right", "center"}
)


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_tuple(value: Any, field_name: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{field_name} must be a tuple, list, or string.")


def _validate_time_range(start_ms: int | None, end_ms: int | None) -> None:
    if start_ms is not None and start_ms < 0:
        raise ValueError("start_ms must be greater than or equal to 0.")
    if end_ms is not None and end_ms < 0:
        raise ValueError("end_ms must be greater than or equal to 0.")
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("start_ms must be less than end_ms.")


@dataclass(frozen=True, slots=True)
class TextIndexDocument:
    """One searchable text document with traceable source identity."""

    doc_id: str
    source_type: TextSourceType
    content: str
    video_id: str
    entity_id: str
    index_schema_version: str
    index_build_id: str
    language: str | None = None
    shot_id: str | None = None
    frame_id: str | None = None
    clip_id: str | None = None
    segment_id: str | None = None
    caption_id: int | None = None
    timestamp_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    title: str | None = None
    description: str | None = None
    keywords: tuple[str, ...] = field(default_factory=tuple)
    regions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _require_non_empty(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_type",
            _require_non_empty(self.source_type, "source_type"),
        )
        if self.source_type not in TEXT_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {self.source_type}.")

        object.__setattr__(
            self,
            "content",
            _require_non_empty(self.content, "content"),
        )
        object.__setattr__(
            self,
            "video_id",
            _require_non_empty(self.video_id, "video_id"),
        )
        object.__setattr__(
            self,
            "entity_id",
            _require_non_empty(self.entity_id, "entity_id"),
        )
        object.__setattr__(
            self,
            "index_schema_version",
            _require_non_empty(self.index_schema_version, "index_schema_version"),
        )
        object.__setattr__(
            self,
            "index_build_id",
            _require_non_empty(self.index_build_id, "index_build_id"),
        )
        object.__setattr__(
            self,
            "keywords",
            _normalize_tuple(self.keywords, "keywords"),
        )
        object.__setattr__(
            self,
            "regions",
            _normalize_tuple(self.regions, "regions"),
        )
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")
        _validate_time_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class TextSearchQuery:
    """Text search request accepted by the Elasticsearch manager."""

    query_text: str
    top_k: int = 50
    from_: int = 0
    page: int = 1
    sort_by: str | None = None
    source_types: tuple[TextSourceType, ...] = field(default_factory=tuple)
    video_ids: tuple[str, ...] = field(default_factory=tuple)
    language: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    ocr_region: str | None = None
    use_fuzzy: bool = True
    use_highlight: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_text",
            _require_non_empty(self.query_text, "query_text"),
        )
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self.page < 1:
            raise ValueError("page must be greater than or equal to 1.")
        if self.from_ < 0:
            raise ValueError("from_ must be greater than or equal to 0.")

        computed_from = self.from_
        if self.page > 1 and self.from_ == 0:
            computed_from = (self.page - 1) * self.top_k
        object.__setattr__(self, "from_", computed_from)

        source_types = _normalize_tuple(self.source_types, "source_types")
        for source_type in source_types:
            if source_type not in TEXT_SOURCE_TYPES:
                raise ValueError(f"Unsupported source_type: {source_type}.")
        object.__setattr__(self, "source_types", source_types)
        object.__setattr__(
            self,
            "video_ids",
            _normalize_tuple(self.video_ids, "video_ids"),
        )

        if self.ocr_region is not None and self.ocr_region not in OCR_REGIONS:
            raise ValueError(f"Unsupported ocr_region: {self.ocr_region}.")
        _validate_time_range(self.start_ms, self.end_ms)


@dataclass(frozen=True, slots=True)
class TextSearchHit:
    """One parsed Elasticsearch search result."""

    doc_id: str
    source_type: TextSourceType
    score: float
    video_id: str
    entity_id: str
    content: str
    highlights: tuple[str, ...] = field(default_factory=tuple)
    shot_id: str | None = None
    frame_id: str | None = None
    clip_id: str | None = None
    segment_id: str | None = None
    caption_id: int | None = None
    timestamp_ms: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _require_non_empty(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_type",
            _require_non_empty(self.source_type, "source_type"),
        )
        if self.source_type not in TEXT_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {self.source_type}.")
        object.__setattr__(
            self,
            "video_id",
            _require_non_empty(self.video_id, "video_id"),
        )
        object.__setattr__(
            self,
            "entity_id",
            _require_non_empty(self.entity_id, "entity_id"),
        )
        object.__setattr__(
            self,
            "content",
            _require_non_empty(self.content, "content"),
        )
        object.__setattr__(
            self,
            "highlights",
            _normalize_tuple(self.highlights, "highlights"),
        )
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")
        _validate_time_range(self.start_ms, self.end_ms)
