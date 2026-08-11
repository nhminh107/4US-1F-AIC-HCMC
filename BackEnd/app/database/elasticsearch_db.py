"""Elasticsearch text-index management for video retrieval evidence."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Callable

from dotenv import load_dotenv

from BackEnd.CONFIG import (
    ELASTICSEARCH_BULK_BATCH_SIZE,
    ELASTICSEARCH_BULK_MAX_BYTES,
    ELASTICSEARCH_INDEX_SCHEMA_VERSION as INDEX_SCHEMA_VERSION,
)
from BackEnd.app.contracts.search import (
    OBJECT_CLASS_SYNONYMS,
    REVERSE_SYNONYMS,
    TextIndexDocument,
    TextSearchHit,
    TextSearchQuery,
    TextSourceType,
)

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk as elasticsearch_bulk
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    Elasticsearch = None  # type: ignore[assignment]
    elasticsearch_bulk = None

load_dotenv()

DEFAULT_SOURCE_ALIASES: dict[TextSourceType, str] = {
    "video_metadata": "aic_hcm2026_text_metadata_active",
    "ocr": "aic_hcm2026_text_ocr_active",
    "transcript": "aic_hcm2026_text_transcript_active",
    "caption": "aic_hcm2026_text_caption_active",
    "object": "aic_hcm2026_text_object_active",
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
    def index_definition(*, bigdata: bool = False) -> dict[str, Any]:
        """Return settings and mappings for the text evidence index.

        Args:
            bigdata: When True, applies settings optimized for bulk ingestion
                     of large datasets (>50K docs): more shards, best_compression,
                     and disabled auto-refresh during indexing.
        """

        text_field = {"type": "text", "analyzer": "aic_vi_text"}
        keyword_field = {"type": "keyword", "norms": False}

        index_settings: dict[str, Any] = {
            "number_of_shards": 3 if bigdata else 1,
            "number_of_replicas": 0 if bigdata else 1,
            "codec": "best_compression" if bigdata else "default",
            "refresh_interval": "-1" if bigdata else "1s",
        }

        return {
            "settings": {
                "index": index_settings,
                "analysis": {
                    "filter": {
                        "aic_ascii_folding": {
                            "type": "asciifolding",
                            "preserve_original": True,
                        },
                        "aic_shingle": {
                            "type": "shingle",
                            "min_shingle_size": 2,
                            "max_shingle_size": 3,
                            "output_unigrams": True,
                        },
                    },
                    "analyzer": {
                        "aic_vi_text": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "aic_ascii_folding",
                            ],
                        },
                        "aic_vi_shingle": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "aic_ascii_folding",
                                "aic_shingle",
                            ],
                        },
                    },
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": keyword_field,
                    "source_type": keyword_field,
                    "video_id": keyword_field,
                    "entity_id": keyword_field,
                    "shot_id": keyword_field,
                    "frame_id": keyword_field,
                    "clip_id": keyword_field,
                    "segment_id": keyword_field,
                    "caption_id": {"type": "long"},
                    "language": keyword_field,
                    "timestamp_ms": {"type": "long"},
                    "start_ms": {"type": "long"},
                    "end_ms": {"type": "long"},
                    "title": text_field,
                    "description": text_field,
                    "keywords": {
                        "type": "text",
                        "analyzer": "aic_vi_text",
                        "fields": {"raw": keyword_field},
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "aic_vi_text",
                        "fields": {
                            "shingle": {
                                "type": "text",
                                "analyzer": "aic_vi_shingle",
                            }
                        },
                    },
                    "ocr_text": text_field,
                    "objects": {
                        "type": "text",
                        "analyzer": "aic_vi_text",
                        "fields": {"raw": keyword_field},
                    },
                    "object_class_ids": keyword_field,
                    "regions": {
                        "type": "nested",
                        "properties": {
                            "n": {"type": "integer"},
                            "text": text_field,
                            "language": keyword_field,
                            "x_min": {"type": "float"},
                            "x_max": {"type": "float"},
                            "y_min": {"type": "float"},
                            "y_max": {"type": "float"},
                        },
                    },
                    "model_name": keyword_field,
                    "model_version": keyword_field,
                    "prompt_version": keyword_field,
                    "index_schema_version": keyword_field,
                    "index_build_id": keyword_field,
                    "indexed_at": {"type": "date"},
                }
            },
        }

    def create_index(self, index_name: str, *, bigdata: bool = False) -> None:
        """Create a physical Elasticsearch index."""

        self.client.indices.create(
            index=index_name, body=self.index_definition(bigdata=bigdata)
        )

    def ensure_index_exists(self, index_name: str, *, bigdata: bool = False) -> bool:
        """Create the index with proper mapping if it doesn't already exist.

        Returns True if the index was created, False if it already existed.
        """
        if self.client.indices.exists(index=index_name):
            logger.info("Index '%s' already exists, skipping creation.", index_name)
            return False
        self.create_index(index_name, bigdata=bigdata)
        logger.info("Created index '%s' (bigdata=%s).", index_name, bigdata)
        return True

    def refresh_index(self, index_name: str) -> None:
        """Explicitly refresh Lucene index segments."""

        self.client.indices.refresh(index=index_name)

    def finalize_bulk_index(self, index_name: str) -> None:
        """Re-enable refresh and force-merge after bulk ingestion.

        Call this after bulk indexing with bigdata=True to restore
        normal search behavior and optimize segments. Automatically detects
        single-node clusters to maintain GREEN health status.
        """
        num_replicas = 0
        try:
            nodes_info = self.client.nodes.stats(metric="indices")
            nodes_dict = nodes_info.get("nodes", {})
            if len(nodes_dict) > 1:
                num_replicas = 1
        except Exception:
            num_replicas = 0

        self.client.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "1s", "number_of_replicas": num_replicas}},
        )
        self.client.indices.refresh(index=index_name)
        try:
            self.client.indices.forcemerge(
                index=index_name, max_num_segments=5, request_timeout=120
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Force merge on '%s' failed (non-fatal): %s", index_name, exc)

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

    def _client_with_options(self) -> Any:
        if hasattr(self.client, "options"):
            return self.client.options(request_timeout=self.request_timeout)
        return self.client

    def index_documents(
        self,
        documents: list[TextIndexDocument],
        *,
        index_name: str | None = None,
        refresh: bool = False,
        chunk_size: int = ELASTICSEARCH_BULK_BATCH_SIZE,
        max_chunk_bytes: int = ELASTICSEARCH_BULK_MAX_BYTES,
    ) -> dict[str, int]:
        """Bulk upsert text documents and return a small result summary."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
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

        # In-memory deduplication per batch
        deduped_docs: dict[str, TextIndexDocument] = {}
        for document in valid_documents:
            if document.doc_id in deduped_docs:
                logger.warning(
                    "Duplicate doc_id '%s' found in batch, keeping latest version.",
                    document.doc_id,
                )
            deduped_docs[document.doc_id] = document
        unique_documents = list(deduped_docs.values())

        total_indexed = 0
        total_failed = 0

        for i in range(0, len(unique_documents), chunk_size):
            chunk = unique_documents[i : i + chunk_size]
            actions = [
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": document.doc_id,
                    "_source": self._document_source(document),
                }
                for document in chunk
            ]

            try:
                client = self._client_with_options()
                success_count, errors = self.bulk_helper(
                    client,
                    actions,
                    refresh=refresh,
                    max_chunk_bytes=max_chunk_bytes,
                    raise_on_error=False,
                )
                total_indexed += int(success_count)
                total_failed += len(errors or [])
            except Exception as error:  # noqa: BLE001 - preserve external helper context.
                raise RuntimeError("Elasticsearch bulk indexing failed.") from error

        return {"indexed": total_indexed, "failed": total_failed}

    def search(self, query: TextSearchQuery) -> list[TextSearchHit]:
        """Search source aliases and parse raw Elasticsearch hits."""

        response = self._client_with_options().search(
            index=self._aliases_for_query(query),
            body=self._search_body(query),
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
        if query.object_region:
            filters.append(self._ocr_region_filter(query.object_region))

        # Tuned boost values for AIC dataset:
        # - match_phrase: 5 (exact phrase match is strongest signal)
        # - title: 5 (video titles are highly descriptive)
        # - objects: 5 (object detection is primary AIC query pattern)
        # - keywords: 4 (curated tags)
        # - content/ocr_text: 3 (body text, moderate signal)
        # - description: 2 (often noisy/long, lower signal)
        # - regions.text: match_phrase^4 + match^2 (handles both exact phrase & word-split OCR)
        should: list[dict[str, Any]] = [
            {"match_phrase": {"content": {"query": query.query_text, "boost": 5}}},
            {
                "multi_match": {
                    "query": query.query_text,
                    "fields": ["title^5", "objects^5", "keywords^4", "content^3", "ocr_text^3", "description^2"],
                    "type": "best_fields",
                }
            },
            {
                "nested": {
                    "path": "regions",
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match_phrase": {
                                        "regions.text": {"query": query.query_text, "boost": 4}
                                    }
                                },
                                {
                                    "match": {
                                        "regions.text": {"query": query.query_text, "boost": 2}
                                    }
                                },
                            ]
                        }
                    },
                }
            },
        ]

        # Bidirectional Vietnamese <-> English Synonym Query Expansion for Objects with Word Boundary Check
        q_lower = query.query_text.lower().strip()
        expanded_terms: list[str] = []
        # VI → EN expansion (using regex \b word boundary to prevent substring matches e.g. "carpet" -> "car")
        for vn_key, en_terms in OBJECT_CLASS_SYNONYMS.items():
            pattern = r"\b" + re.escape(vn_key) + r"\b"
            if re.search(pattern, q_lower):
                expanded_terms.extend(en_terms)
        # EN → VI expansion (reverse lookup with word boundary)
        for en_key, vn_terms in REVERSE_SYNONYMS.items():
            pattern = r"\b" + re.escape(en_key) + r"\b"
            if re.search(pattern, q_lower):
                expanded_terms.extend(vn_terms)
        if expanded_terms:
            synonym_query = " ".join(set(expanded_terms))
            should.append({
                "multi_match": {
                    "query": synonym_query,
                    "fields": ["objects^5", "content^3"],
                    "type": "best_fields",
                }
            })

        if query.source_boosts:
            for stype, boost_val in query.source_boosts.items():
                should.append({
                    "term": {
                        "source_type": {
                            "value": stype,
                            "boost": boost_val,
                        }
                    }
                })

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
            "from": query.from_,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
        }
        if query.sort_by:
            if query.sort_by == "score":
                body["sort"] = ["_score"]
            elif query.sort_by.startswith("-"):
                field_name = query.sort_by[1:]
                body["sort"] = [{field_name: "desc"}]
            else:
                body["sort"] = [{query.sort_by: "asc"}]
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
                {"range": {"regions.x_min": {"gte": 0.20}}},
                {"range": {"regions.x_max": {"lte": 0.80}}},
                {"range": {"regions.y_min": {"gte": 0.20}}},
                {"range": {"regions.y_max": {"lte": 0.80}}},
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
            objects=tuple(source.get("objects") or ()),
            shot_id=source.get("shot_id"),
            frame_id=source.get("frame_id"),
            clip_id=source.get("clip_id"),
            segment_id=source.get("segment_id"),
            caption_id=source.get("caption_id"),
            timestamp_ms=source.get("timestamp_ms"),
            start_ms=source.get("start_ms"),
            end_ms=source.get("end_ms"),
        )
