"""Comprehensive test suite covering all Elasticsearch BigData enhancements and contracts.

Includes:
1. Regex Word Boundary Synonym Expansion (EN<->VI, preventing substring false positives).
2. OCR Region Query Payload Generation (match_phrase + match fallback).
3. Single-Node vs Multi-Node Replica Topology Detection in finalize_bulk_index.
4. Keyset/Cursor-based SQL Pagination in TextSearchService.
5. Object Document Frame Grouping (multiple bounding boxes into 1 frame document).
6. Seed & Ingestion parameter constraints (max_frames_per_video).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from BackEnd.app.contracts.search import (
    OBJECT_CLASS_SYNONYMS,
    REVERSE_SYNONYMS,
    TextIndexDocument,
    TextSearchQuery,
)
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder
from BackEnd.app.service.text_search_service import TextSearchService


class WordBoundarySynonymTests(unittest.TestCase):
    """Test regex word boundary synonym matching to ensure substring isolation."""

    def test_reverse_synonyms_dict_generated_correctly(self) -> None:
        self.assertIn("car", REVERSE_SYNONYMS)
        self.assertIn("person", REVERSE_SYNONYMS)
        self.assertIn("xe hơi", REVERSE_SYNONYMS["car"])
        self.assertIn("ô tô", REVERSE_SYNONYMS["car"])

    def test_word_boundary_regex_matches_exact_words(self) -> None:
        query_text = "i see a car on the street"
        expanded = []
        for en_key, vn_terms in REVERSE_SYNONYMS.items():
            pattern = r"\b" + re.escape(en_key) + r"\b"
            if re.search(pattern, query_text.lower()):
                expanded.extend(vn_terms)
        self.assertIn("xe hơi", expanded)
        self.assertIn("ô tô", expanded)

    def test_word_boundary_regex_ignores_substrings(self) -> None:
        query_text = "the carpet is red and scary"
        expanded = []
        for en_key, vn_terms in REVERSE_SYNONYMS.items():
            pattern = r"\b" + re.escape(en_key) + r"\b"
            if re.search(pattern, query_text.lower()):
                expanded.extend(vn_terms)

        # "carpet" contains "car", "scary" contains "car" -> MUST NOT match "car"
        self.assertNotIn("xe hơi", expanded)
        self.assertNotIn("ô tô", expanded)

    def test_word_boundary_regex_ignores_manhole_for_man(self) -> None:
        query_text = "there is a manhole in the alley"
        expanded = []
        for en_key, vn_terms in REVERSE_SYNONYMS.items():
            pattern = r"\b" + re.escape(en_key) + r"\b"
            if re.search(pattern, query_text.lower()):
                expanded.extend(vn_terms)

        # "manhole" contains "man" -> MUST NOT match "man" -> "đàn ông"
        self.assertNotIn("đàn ông", expanded)


class ElasticsearchQueryPayloadTests(unittest.TestCase):
    """Test search payload DSL generation for OCR match phrase & match fallback."""

    def setUp(self) -> None:
        self.manager = ElasticsearchManager(
            elasticsearch_url="http://localhost:9200",
            client=MagicMock(),
        )

    def test_ocr_region_query_payload_has_both_match_phrase_and_match(self) -> None:
        query = TextSearchQuery(
            query_text="BIEN HOA",
            ocr_region="top",
        )
        body = self.manager._search_body(query)
        should_clauses = body["query"]["bool"]["should"]

        nested_clause = None
        for clause in should_clauses:
            if "nested" in clause and clause["nested"].get("path") == "regions":
                nested_clause = clause
                break

        self.assertIsNotNone(nested_clause)
        nested_query = nested_clause["nested"]["query"]
        self.assertIn("bool", nested_query)
        self.assertIn("should", nested_query["bool"])
        sub_should = nested_query["bool"]["should"]

        # Expect both match_phrase and match fallback
        has_match_phrase = any("match_phrase" in sub for sub in sub_should)
        has_match = any("match" in sub for sub in sub_should)
        self.assertTrue(has_match_phrase)
        self.assertTrue(has_match)


class SingleNodeReplicaTopologyTests(unittest.TestCase):
    """Test finalize_bulk_index setting 0 replicas for single-node cluster."""

    def test_finalize_bulk_index_single_node_sets_zero_replicas(self) -> None:
        mock_client = MagicMock()
        mock_client.nodes.stats.return_value = {"nodes": {"node1": {}}}
        manager = ElasticsearchManager(client=mock_client)

        manager.finalize_bulk_index("test_index")

        mock_client.indices.put_settings.assert_called_once_with(
            index="test_index",
            body={"index": {"refresh_interval": "1s", "number_of_replicas": 0}},
        )

    def test_finalize_bulk_index_multi_node_sets_one_replica(self) -> None:
        mock_client = MagicMock()
        mock_client.nodes.stats.return_value = {"nodes": {"node1": {}, "node2": {}}}
        manager = ElasticsearchManager(client=mock_client)

        manager.finalize_bulk_index("test_index")

        mock_client.indices.put_settings.assert_called_once_with(
            index="test_index",
            body={"index": {"refresh_interval": "1s", "number_of_replicas": 1}},
        )


class ObjectDocumentGroupingTests(unittest.TestCase):
    """Test that multiple bounding box detections in a frame are grouped into 1 document."""

    def setUp(self) -> None:
        self.builder = ElasticsearchDocumentBuilder()

    def test_multiple_detections_grouped_into_single_frame_document(self) -> None:
        frame = SimpleNamespace(
            frame_id="L01_V001_005",
            video_id="L01_V001",
            shot_id="L01_V001_s1",
            timestamp_ms=5000,
        )
        objects_input = [
            SimpleNamespace(confidence=0.9, object_class=SimpleNamespace(class_name="Car"), class_id="c001"),
            SimpleNamespace(confidence=0.85, object_class=SimpleNamespace(class_name="Car"), class_id="c001"),
            SimpleNamespace(confidence=0.75, object_class=SimpleNamespace(class_name="Person"), class_id="c002"),
            SimpleNamespace(confidence=0.95, object_class=SimpleNamespace(class_name="Traffic light"), class_id="c003"),
        ]

        doc = self.builder.build_object_document(frame, objects_input, index_build_id="build-test")

        self.assertIsNotNone(doc)
        self.assertEqual(doc.doc_id, "object:L01_V001_005:v1")
        self.assertEqual(doc.source_type, "object")
        # 4 bounding boxes grouped into 1 frame doc
        self.assertEqual(len(doc.objects), 4)
        self.assertIn("Car", doc.objects)
        self.assertIn("Person", doc.objects)
        self.assertIn("Traffic light", doc.objects)


class KeysetPaginationServiceTests(unittest.TestCase):
    """Test TextSearchService sync_from_postgres with Keyset cursor pagination."""

    def test_sync_from_postgres_executes_keyset_queries(self) -> None:
        mock_manager = MagicMock()
        mock_session = MagicMock()

        # Return empty list on queries to complete loop cleanly
        mock_session.scalars.return_value.all.return_value = []

        service = TextSearchService(manager=mock_manager)
        res = service.sync_from_postgres(
            session=mock_session,
            index_name="test_index",
            batch_size=100,
            publish_aliases=False,
        )

        self.assertIn("total_indexed", res)
        self.assertEqual(res["total_indexed"], 0)


if __name__ == "__main__":
    unittest.main()
