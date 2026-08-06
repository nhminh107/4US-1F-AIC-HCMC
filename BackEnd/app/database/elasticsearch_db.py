"""Elasticsearch text-index management for video retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import os
from typing import Any, Callable

from dotenv import load_dotenv

from BackEnd.app.contracts.search import (
    TextIndexDocument,
    TextSearchHit,
    TextSearchQuery,
    TextSourceType,
)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk as elasticsearch_bulk
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    Elasticsearch = None  # type: ignore[assignment]
    elasticsearch_bulk = None

load_dotenv()

INDEX_SCHEMA_VERSION = "text-index-schema@1.0.0"
DEFAULT_SOURCE_ALIASES: dict[TextSourceType, str] = {
    "video_metadata": "aic_hcm2026_text_metadata_active",
    "ocr": "aic_hcm2026_text_ocr_active",
    "transcript": "aic_hcm2026_text_transcript_active",
    "caption": "aic_hcm2026_text_caption_active",
}


class ElasticsearchManager:
    """Create, populate, and query the AIC text evidence index."""

    def __init__(
        self,
        elasticsearch_url: str | None = None,
        *,
        source_aliases: dict[TextSourceType, str] | None = None,
        request_timeout: int = 30,
        client: Any | None = None,
        bulk_helper: Callable | None = None,
    ) -> None:
        self.source_aliases = dict(
            source_aliases if source_aliases is not None else DEFAULT_SOURCE_ALIASES
        )
        self.request_timeout = request_timeout
        self.bulk_helper = bulk_helper or elasticsearch_bulk

        if client is not None:
            self.client = client
            return

        if Elasticsearch is None:
            raise RuntimeError("The elasticsearch package is not installed.")

        resolved_url = elasticsearch_url or os.getenv("ELASTICSEARCH_URL")
        if not resolved_url:
            raise RuntimeError(
                "ELASTICSEARCH_URL is not configured. "
                "Set it in the environment or pass elasticsearch_url explicitly."
            )
        self.client = Elasticsearch(resolved_url, request_timeout=request_timeout)

    @staticmethod
    def index_definition() -> dict[str, Any]:
        """Return settings and mappings for the text evidence index."""

        text_field = {"type": "text", "analyzer": "aic_vi_text"}
        return {
            "settings": {
                "analysis": {
                    "filter": {
                        "aic_ascii_folding": {
                            "type": "asciifolding",
                            "preserve_original": True,
                        }
                    },
                    "analyzer": {
                        "aic_vi_text": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "aic_ascii_folding"],
                        }
                    },
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "video_id": {"type": "keyword"},
                    "entity_id": {"type": "keyword"},
                    "shot_id": {"type": "keyword"},
                    "frame_id": {"type": "keyword"},
                    "clip_id": {"type": "keyword"},
                    "segment_id": {"type": "keyword"},
                    "caption_id": {"type": "long"},
                    "language": {"type": "keyword"},
                    "timestamp_ms": {"type": "long"},
                    "start_ms": {"type": "long"},
                    "end_ms": {"type": "long"},
                    "title": text_field,
                    "description": text_field,
                    "keywords": {
                        "type": "text",
                        "analyzer": "aic_vi_text",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "content": text_field,
                    "regions": {
                        "type": "nested",
                        "properties": {
                            "n": {"type": "integer"},
                            "text": text_field,
                            "language": {"type": "keyword"},
                            "x_min": {"type": "float"},
                            "x_max": {"type": "float"},
                            "y_min": {"type": "float"},
                            "y_max": {"type": "float"},
                        },
                    },
                    "model_name": {"type": "keyword"},
                    "model_version": {"type": "keyword"},
                    "prompt_version": {"type": "keyword"},
                    "index_schema_version": {"type": "keyword"},
                    "index_build_id": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
        }

    def create_index(self, index_name: str) -> None:
        """Create a physical Elasticsearch index."""

        self.client.indices.create(index=index_name, body=self.index_definition())

    def publish_source_aliases(self, index_name: str) -> None:
        """Atomically point source-specific filtered aliases at one index."""

        actions = []
        for source_type, alias in self.source_aliases.items():
            actions.append(
                {
                    "remove": {
                        "index": "*",
                        "alias": alias,
                        "must_exist": False,
                    }
                }
            )
            actions.append(
                {
                    "add": {
                        "index": index_name,
                        "alias": alias,
                        "filter": {"term": {"source_type": source_type}},
                    }
                }
            )

        response = self.client.indices.update_aliases(body={"actions": actions})
        self._raise_for_unexpected_alias_errors(response)

    def index_documents(
        self,
        documents: list[TextIndexDocument],
        *,
        index_name: str | None = None,
        refresh: bool = False,
    ) -> dict[str, int]:
        """Bulk upsert text documents and return a small result summary."""

        valid_documents = [doc for doc in (documents or []) if doc is not None]
        if not valid_documents:
            return {"indexed": 0, "failed": 0}
        if self.bulk_helper is None:
            raise RuntimeError("Elasticsearch bulk helper is not available.")
        if not index_name:
            raise ValueError(
                "index_name is required when indexing documents. "
                "Use the versioned physical index, not a source alias."
            )

        doc_ids = [document.doc_id for document in valid_documents]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Duplicate doc_id values are not allowed in one batch.")

        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": document.doc_id,
                "_source": self._document_source(document),
            }
            for document in valid_documents
        ]

        try:
            success_count, errors = self.bulk_helper(
                self.client,
                actions,
                refresh=refresh,
                raise_on_error=False,
                request_timeout=self.request_timeout,
            )
        except Exception as error:  # noqa: BLE001 - preserve external helper context.
            raise RuntimeError("Elasticsearch bulk indexing failed.") from error

        return {"indexed": int(success_count), "failed": len(errors or [])}

    def search(self, query: TextSearchQuery) -> list[TextSearchHit]:
        """Search source aliases and parse raw Elasticsearch hits."""

        response = self.client.search(
            index=self._aliases_for_query(query),
            body=self._search_body(query),
            request_timeout=self.request_timeout,
        )
        return [self._parse_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def health_check(self, index_name: str | None = None) -> dict[str, object]:
        """Return basic Elasticsearch connectivity and alias information."""

        info = self.client.info()
        aliases = None
        if index_name is not None:
            aliases = self.client.indices.get_alias(index=index_name)
        return {"info": info, "aliases": aliases}

    def _aliases_for_query(self, query: TextSearchQuery) -> list[str]:
        source_types = query.source_types or tuple(self.source_aliases)
        return [self._alias_for_source(source_type) for source_type in source_types]

    def _alias_for_source(self, source_type: TextSourceType) -> str:
        try:
            return self.source_aliases[source_type]
        except KeyError as error:
            raise ValueError(f"No alias configured for source_type={source_type}.") from error

    @staticmethod
    def _raise_for_unexpected_alias_errors(response: dict[str, Any] | None) -> None:
        """Raise when the aliases API reports a non-ignorable action error."""

        if not response or not response.get("errors"):
            return

        unexpected_errors = []
        for result in response.get("action_results", []):
            error = result.get("error")
            if not error:
                continue
            action_type = result.get("action", {}).get("type")
            error_type = error.get("type")
            if action_type == "remove" and error_type in (
                "aliases_not_found_exception",
                "index_not_found_exception",
            ):
                continue
            unexpected_errors.append(result)

        if unexpected_errors:
            raise RuntimeError(
                f"Unexpected Elasticsearch alias update errors: {unexpected_errors}"
            )

    @staticmethod
    def _document_source(document: TextIndexDocument) -> dict[str, Any]:
        source = asdict(document)
        source["indexed_at"] = datetime.now(UTC).isoformat()
        return source

    def _search_body(self, query: TextSearchQuery) -> dict[str, Any]:
        filters: list[dict[str, Any]] = []
        if query.source_types:
            filters.append({"terms": {"source_type": list(query.source_types)}})
        if query.video_ids:
            filters.append({"terms": {"video_id": list(query.video_ids)}})
        if query.language:
            filters.append({"term": {"language": query.language}})
        filters.extend(self._temporal_filters(query))
        if query.ocr_region:
            filters.append(self._ocr_region_filter(query.ocr_region))

        should: list[dict[str, Any]] = [
            {"match_phrase": {"content": {"query": query.query_text, "boost": 6}}},
            {
                "multi_match": {
                    "query": query.query_text,
                    "fields": ["title^5", "keywords^4", "content^3", "description"],
                    "type": "best_fields",
                }
            },
            {
                "nested": {
                    "path": "regions",
                    "query": {
                        "match_phrase": {
                            "regions.text": {"query": query.query_text, "boost": 3}
                        }
                    },
                }
            },
        ]
        if query.use_fuzzy:
            should.append(
                {
                    "match": {
                        "content": {
                            "query": query.query_text,
                            "fuzziness": "AUTO",
                            "boost": 1,
                        }
                    }
                }
            )

        body: dict[str, Any] = {
            "size": query.top_k,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
        }
        if query.use_highlight:
            body["highlight"] = {"fields": {"content": {}, "regions.text": {}}}
        return body

    @staticmethod
    def _temporal_filters(query: TextSearchQuery) -> list[dict[str, Any]]:
        if query.start_ms is None and query.end_ms is None:
            return []

        should: list[dict[str, Any]] = []
        if query.end_ms is not None:
            should.append({"range": {"start_ms": {"lt": query.end_ms}}})
        if query.start_ms is not None:
            should.append({"range": {"end_ms": {"gt": query.start_ms}}})

        timestamp_filters: list[dict[str, Any]] = []
        if query.start_ms is not None:
            timestamp_filters.append({"range": {"timestamp_ms": {"gte": query.start_ms}}})
        if query.end_ms is not None:
            timestamp_filters.append({"range": {"timestamp_ms": {"lt": query.end_ms}}})

        return [
            {
                "bool": {
                    "should": [
                        {"bool": {"filter": should}},
                        {"bool": {"filter": timestamp_filters}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        ]

    @staticmethod
    def _ocr_region_filter(region: str) -> dict[str, Any]:
        region_filters: dict[str, list[dict[str, Any]]] = {
            "top": [{"range": {"regions.y_max": {"lte": 0.25}}}],
            "header": [{"range": {"regions.y_max": {"lte": 0.25}}}],
            "bottom": [{"range": {"regions.y_min": {"gte": 0.75}}}],
            "footer": [{"range": {"regions.y_min": {"gte": 0.75}}}],
            "left": [{"range": {"regions.x_max": {"lte": 0.35}}}],
            "right": [{"range": {"regions.x_min": {"gte": 0.65}}}],
            "center": [
                {"range": {"regions.x_min": {"lte": 0.65}}},
                {"range": {"regions.x_max": {"gte": 0.35}}},
                {"range": {"regions.y_min": {"lte": 0.65}}},
                {"range": {"regions.y_max": {"gte": 0.35}}},
            ],
        }
        return {
            "nested": {
                "path": "regions",
                "query": {"bool": {"filter": region_filters[region]}},
            }
        }

    @staticmethod
    def _parse_hit(hit: dict[str, Any]) -> TextSearchHit:
        source = hit.get("_source") or {}
        highlights = []
        for values in (hit.get("highlight") or {}).values():
            highlights.extend(values)
        return TextSearchHit(
            doc_id=source.get("doc_id") or hit.get("_id"),
            source_type=source.get("source_type"),
            score=float(hit.get("_score") or 0.0),
            video_id=source.get("video_id"),
            entity_id=source.get("entity_id"),
            content=source.get("content"),
            highlights=tuple(highlights),
            shot_id=source.get("shot_id"),
            frame_id=source.get("frame_id"),
            clip_id=source.get("clip_id"),
            segment_id=source.get("segment_id"),
            caption_id=source.get("caption_id"),
            timestamp_ms=source.get("timestamp_ms"),
            start_ms=source.get("start_ms"),
            end_ms=source.get("end_ms"),
        )
