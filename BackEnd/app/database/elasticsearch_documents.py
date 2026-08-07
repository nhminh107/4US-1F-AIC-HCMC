"""Build Elasticsearch text documents from PostgreSQL ORM records."""

from __future__ import annotations

from typing import Any

from BackEnd.app.contracts.search import TextIndexDocument
from BackEnd.app.database.elasticsearch_db import INDEX_SCHEMA_VERSION


class ElasticsearchDocumentBuilder:
    """Convert PostgreSQL source records into text-index contracts."""

    def build_video_metadata_document(
        self,
        video: Any,
        *,
        index_build_id: str,
    ) -> TextIndexDocument | None:
        raw_keywords = getattr(video, "keywords", None)
        if isinstance(raw_keywords, str):
            keywords = tuple(k.strip() for k in raw_keywords.split(",") if k.strip())
        else:
            keywords = tuple(raw_keywords or ())

        content = self._join_text(
            getattr(video, "title", None),
            getattr(video, "description", None),
            " ".join(keywords),
        )
        if not content:
            return None

        video_id = str(getattr(video, "video_id"))
        return TextIndexDocument(
            doc_id=f"video_metadata:{video_id}:v1",
            source_type="video_metadata",
            content=content,
            video_id=video_id,
            entity_id=video_id,
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=index_build_id,
            title=getattr(video, "title", None),
            description=getattr(video, "description", None),
            keywords=keywords,
        )

    def build_ocr_document(
        self,
        frame: Any,
        ocr_records: list[Any],
        *,
        index_build_id: str,
    ) -> TextIndexDocument | None:
        sorted_records = sorted(ocr_records, key=lambda record: getattr(record, "n"))
        regions: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for record in sorted_records:
            text = self._normalize_text(getattr(record, "text", None))
            if not text:
                continue
            try:
                self._validate_box(record)
            except (ValueError, TypeError, AttributeError):
                continue
            text_parts.append(text)
            regions.append(
                {
                    "n": getattr(record, "n"),
                    "text": text,
                    "language": getattr(record, "language", None),
                    "x_min": getattr(record, "x_min"),
                    "x_max": getattr(record, "x_max"),
                    "y_min": getattr(record, "y_min"),
                    "y_max": getattr(record, "y_max"),
                }
            )

        content = self._join_text(*text_parts)
        if not content:
            return None

        frame_id = str(getattr(frame, "frame_id"))
        return TextIndexDocument(
            doc_id=f"ocr_frame:{frame_id}:v1",
            source_type="ocr",
            content=content,
            video_id=str(getattr(frame, "video_id")),
            entity_id=frame_id,
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=index_build_id,
            language=self._first_non_empty(region["language"] for region in regions),
            shot_id=getattr(frame, "shot_id", None),
            frame_id=frame_id,
            timestamp_ms=getattr(frame, "timestamp_ms", None),
            regions=tuple(regions),
        )

    def build_transcript_document(
        self,
        segment: Any,
        *,
        index_build_id: str,
    ) -> TextIndexDocument | None:
        content = self._normalize_text(getattr(segment, "text", None))
        if not content:
            return None

        segment_id = str(getattr(segment, "segment_id"))
        return TextIndexDocument(
            doc_id=f"transcript:{segment_id}:v1",
            source_type="transcript",
            content=content,
            video_id=str(getattr(segment, "video_id")),
            entity_id=segment_id,
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=index_build_id,
            language=getattr(segment, "language", None),
            segment_id=segment_id,
            start_ms=getattr(segment, "start_ms", None),
            end_ms=getattr(segment, "end_ms", None),
        )

    def build_caption_document(
        self,
        caption: Any,
        *,
        index_build_id: str,
        video_id: str | None = None,
    ) -> TextIndexDocument | None:
        content = self._normalize_text(getattr(caption, "caption_text", None))
        if not content:
            return None

        caption_id = getattr(caption, "caption_id")
        resolved_video_id = video_id or self._resolve_caption_video_id(caption)
        return TextIndexDocument(
            doc_id=f"caption:{caption_id}:v1",
            source_type="caption",
            content=content,
            video_id=str(resolved_video_id),
            entity_id=str(caption_id),
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=index_build_id,
            frame_id=getattr(caption, "frame_id", None),
            clip_id=getattr(caption, "clip_id", None),
            shot_id=getattr(caption, "shot_id", None),
            caption_id=caption_id,
            model_name=getattr(caption, "model_name", None),
            model_version=getattr(caption, "model_version", None),
            prompt_version=getattr(caption, "prompt_version", None),
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _join_text(self, *values: Any) -> str:
        return " ".join(
            text
            for text in (self._normalize_text(value) for value in values)
            if text
        )

    @staticmethod
    def _validate_box(record: Any) -> None:
        x_min = float(getattr(record, "x_min"))
        x_max = float(getattr(record, "x_max"))
        y_min = float(getattr(record, "y_min"))
        y_max = float(getattr(record, "y_max"))
        if not (0 <= x_min <= x_max <= 1 and 0 <= y_min <= y_max <= 1):
            raise ValueError("Invalid OCR bounding box.")

    @staticmethod
    def _first_non_empty(values) -> str | None:
        for value in values:
            if value:
                return value
        return None

    @staticmethod
    def _resolve_caption_video_id(caption: Any) -> str:
        frame = getattr(caption, "frame", None)
        if frame is not None and getattr(frame, "video_id", None):
            return str(frame.video_id)

        shot = getattr(caption, "shot", None)
        if shot is not None and getattr(shot, "video_id", None):
            return str(shot.video_id)

        clip = getattr(caption, "clip", None)
        clip_shot = getattr(clip, "shot", None)
        if clip_shot is not None and getattr(clip_shot, "video_id", None):
            return str(clip_shot.video_id)

        raise ValueError("caption video_id could not be resolved.")
