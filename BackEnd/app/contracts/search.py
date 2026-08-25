"""Typed contracts for Elasticsearch text indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TextSourceType = Literal["video_metadata", "ocr", "transcript", "caption", "object"]

TEXT_SOURCE_TYPES: frozenset[str] = frozenset(
    {"video_metadata", "ocr", "transcript", "caption", "object"}
)
OCR_REGIONS: frozenset[str] = frozenset(
    {"top", "header", "bottom", "footer", "left", "right", "center"}
)
OBJECT_REGIONS: frozenset[str] = OCR_REGIONS

# Mapping Vietnamese search query terms to English OpenImages/COCO object classes
OBJECT_CLASS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "xe hơi": ("Car", "Vehicle", "Land vehicle", "Automobile"),
    "ô tô": ("Car", "Vehicle", "Land vehicle", "Automobile"),
    "người": ("Person", "Man", "Woman", "Boy", "Girl", "Human face", "Human body", "Human head"),
    "con người": ("Person", "Man", "Woman", "Boy", "Girl"),
    "đàn ông": ("Man", "Person"),
    "phụ nữ": ("Woman", "Girl", "Person"),
    "con gái": ("Girl", "Woman", "Person"),
    "con trai": ("Boy", "Man", "Person"),
    "tòa nhà": ("Building", "Skyscraper", "House"),
    "nhà": ("Building", "House"),
    "xe máy": ("Motorcycle", "Vehicle", "Land vehicle"),
    "xe đạp": ("Bicycle", "Vehicle", "Land vehicle"),
    "mặt người": ("Human face",),
    "khuôn mặt": ("Human face",),
    "mắt kính": ("Glasses", "Fashion accessory"),
    "kính": ("Glasses",),
    "quần áo": ("Clothing", "Shirt", "Dress", "Suit", "Trousers"),
    "áo": ("Shirt", "Clothing", "Dress", "Suit"),
    "váy": ("Dress", "Clothing"),
    "bàn": ("Table", "Desk", "Furniture"),
    "ghế": ("Chair", "Furniture"),
    "con chó": ("Dog", "Mammal", "Animal"),
    "con mèo": ("Cat", "Mammal", "Animal"),
    "tháp": ("Tower", "Building", "Skyscraper"),
    "poster": ("Poster",),
    "áp phích": ("Poster",),
}

# Auto-generated reverse map: English object class name → Vietnamese search terms
# Enables bidirectional synonym expansion (both VI→EN and EN→VI)
_reverse: dict[str, list[str]] = {}
for _vn_key, _en_values in OBJECT_CLASS_SYNONYMS.items():
    for _en_val in _en_values:
        _en_lower = _en_val.lower()
        if _en_lower not in _reverse:
            _reverse[_en_lower] = []
        if _vn_key not in _reverse[_en_lower]:
            _reverse[_en_lower].append(_vn_key)
REVERSE_SYNONYMS: dict[str, tuple[str, ...]] = {
    k: tuple(v) for k, v in _reverse.items()
}


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
    ocr_text: str | None = None
    keywords: tuple[str, ...] = field(default_factory=tuple)
    objects: tuple[str, ...] = field(default_factory=tuple)
    object_class_ids: tuple[str, ...] = field(default_factory=tuple)
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
            "objects",
            _normalize_tuple(self.objects, "objects"),
        )
        object.__setattr__(
            self,
            "object_class_ids",
            _normalize_tuple(self.object_class_ids, "object_class_ids"),
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
    object_region: str | None = None
    source_boosts: dict[TextSourceType, float] | None = None
    use_fuzzy: bool = False
    use_highlight: bool = False

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

        if self.from_ + self.top_k > 10_000:
            raise ValueError("from_ + top_k must not exceed 10000.")

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
        if self.object_region is not None and self.object_region not in OBJECT_REGIONS:
            raise ValueError(f"Unsupported object_region: {self.object_region}.")
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
    objects: tuple[str, ...] = field(default_factory=tuple)
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
        object.__setattr__(
            self,
            "objects",
            _normalize_tuple(self.objects, "objects"),
        )
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be greater than or equal to 0.")
        _validate_time_range(self.start_ms, self.end_ms)
