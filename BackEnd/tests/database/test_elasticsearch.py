"""Unit tests for Elasticsearch text-index contracts and manager behavior."""

from __future__ import annotations

import importlib
import os
import unittest
from uuid import uuid4

from BackEnd.app.contracts.search import (
    TextIndexDocument,
    TextSearchQuery,
    TextSourceType,
)
from BackEnd.app.database.elasticsearch_db import (
    DEFAULT_SOURCE_ALIASES,
    ElasticsearchManager,
)
from BackEnd.app.database.demo_elasticsearch import build_sample_documents


class FakeIndicesClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.alias_updates: list[dict] = []
        self.deleted: list[dict] = []

    def create(self, **kwargs) -> dict:
        self.created.append(kwargs)
        return {"acknowledged": True}

    def update_aliases(self, **kwargs) -> dict:
        self.alias_updates.append(kwargs)
        return {"acknowledged": True}

    def get_alias(self, **kwargs) -> dict:
        return {"aliases": kwargs}

    def delete(self, **kwargs) -> dict:
        self.deleted.append(kwargs)
        return {"acknowledged": True}


class FakeElasticsearchClient:
    def __init__(self, *, search_response: dict | None = None) -> None:
        self.indices = FakeIndicesClient()
        self.search_calls: list[dict] = []
        self.info_called = False
        self.search_response = search_response or {"hits": {"hits": []}}

    def info(self) -> dict:
        self.info_called = True
        return {"cluster_name": "fake-es"}

    def search(self, **kwargs) -> dict:
        self.search_calls.append(kwargs)
        return self.search_response


def make_document(
    *,
    doc_id: str = "ocr_frame:L21_V001_001:v1",
    source_type: TextSourceType = "ocr",
    content: str = "Lễ trao giải AIC",
    entity_id: str = "L21_V001_001",
) -> TextIndexDocument:
    return TextIndexDocument(
        doc_id=doc_id,
        source_type=source_type,
        content=content,
        video_id="L21_V001",
        entity_id=entity_id,
        index_schema_version="text-index-schema@1.0.0",
        index_build_id="text-index-test-build",
        frame_id="L21_V001_001" if source_type == "ocr" else None,
        timestamp_ms=0 if source_type == "ocr" else None,
    )


class TextSearchContractTests(unittest.TestCase):
    def test_text_index_document_rejects_missing_canonical_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "doc_id"):
            TextIndexDocument(
                doc_id="",
                source_type="ocr",
                content="text",
                video_id="L21_V001",
                entity_id="L21_V001_001",
                index_schema_version="text-index-schema@1.0.0",
                index_build_id="build",
            )

    def test_text_search_query_validates_text_top_k_time_and_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_text"):
            TextSearchQuery(query_text="   ")
        with self.assertRaisesRegex(ValueError, "top_k"):
            TextSearchQuery(query_text="text", top_k=0)
        with self.assertRaisesRegex(ValueError, "start_ms"):
            TextSearchQuery(query_text="text", start_ms=10, end_ms=10)
        with self.assertRaisesRegex(ValueError, "ocr_region"):
            TextSearchQuery(query_text="text", ocr_region="diagonal")

    def test_tuple_fields_are_normalized_without_mutating_caller_lists(self) -> None:
        keywords = ["AIC", "AIC", "HCMC"]
        document = TextIndexDocument(
            doc_id="video_metadata:L21_V001:v1",
            source_type="video_metadata",
            content="AIC HCMC",
            video_id="L21_V001",
            entity_id="L21_V001",
            index_schema_version="text-index-schema@1.0.0",
            index_build_id="build",
            keywords=keywords,  # type: ignore[arg-type]
        )

        self.assertEqual(document.keywords, ("AIC", "AIC", "HCMC"))
        self.assertEqual(keywords, ["AIC", "AIC", "HCMC"])

    def test_text_search_query_pagination_and_sorting(self) -> None:
        query_default = TextSearchQuery(query_text="test")
        self.assertEqual(query_default.from_, 0)
        self.assertEqual(query_default.page, 1)

        query_page = TextSearchQuery(query_text="test", top_k=20, page=3)
        self.assertEqual(query_page.from_, 40)

        with self.assertRaisesRegex(ValueError, "page"):
            TextSearchQuery(query_text="test", page=0)
        with self.assertRaisesRegex(ValueError, "from_"):
            TextSearchQuery(query_text="test", from_=-1)

    def test_text_search_query_boundary_values(self) -> None:
        query_min = TextSearchQuery(query_text="min", top_k=1, from_=0, sort_by="score")
        self.assertEqual(query_min.top_k, 1)
        self.assertEqual(query_min.from_, 0)

        query_max = TextSearchQuery(query_text="max", top_k=5000, from_=5000, sort_by="-timestamp_ms")
        self.assertEqual(query_max.top_k, 5000)
        self.assertEqual(query_max.from_, 5000)

        with self.assertRaisesRegex(ValueError, r"from_ \+ top_k must not exceed 10000."):
            TextSearchQuery(query_text="exceed", top_k=6000, from_=5000)

    def test_text_search_query_multiline_whitespace_and_newlines_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_text"):
            TextSearchQuery(query_text="  \n \t  ")

    def test_text_search_query_normalizes_single_string_inputs(self) -> None:
        query = TextSearchQuery(
            query_text="le trao giai",
            source_types="ocr",  # type: ignore[arg-type]
            video_ids="L21_V001",  # type: ignore[arg-type]
        )
        self.assertEqual(query.source_types, ("ocr",))
        self.assertEqual(query.video_ids, ("L21_V001",))


class ElasticsearchManagerTests(unittest.TestCase):
    def test_index_documents_invalid_chunk_size_raises_value_error(self) -> None:
        manager = ElasticsearchManager(client=FakeElasticsearchClient())
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            manager.index_documents([make_document()], index_name="test", chunk_size=0)

    def test_search_sort_payload_generation(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.search(TextSearchQuery(query_text="test", sort_by="score"))
        self.assertEqual(client.search_calls[-1]["body"]["sort"], ["_score"])

        manager.search(TextSearchQuery(query_text="test", sort_by="-timestamp_ms"))
        self.assertEqual(client.search_calls[-1]["body"]["sort"], [{"timestamp_ms": "desc"}])

        manager.search(TextSearchQuery(query_text="test", sort_by="video_id"))
        self.assertEqual(client.search_calls[-1]["body"]["sort"], [{"video_id": "asc"}])
    def test_create_index_contains_required_mapping_and_analyzer(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.create_index("aic_hcm2026_text_v1_test")

        body = client.indices.created[0]["body"]
        self.assertIn("aic_vi_text", body["settings"]["analysis"]["analyzer"])
        self.assertIn("aic_shingle", body["settings"]["analysis"]["filter"])
        properties = body["mappings"]["properties"]
        for field_name in (
            "doc_id",
            "source_type",
            "video_id",
            "entity_id",
            "content",
            "regions",
            "index_build_id",
        ):
            self.assertIn(field_name, properties)
        self.assertEqual(properties["regions"]["type"], "nested")
        self.assertEqual(properties["keywords"]["type"], "text")
        self.assertEqual(properties["keywords"]["fields"]["raw"]["type"], "keyword")

    def test_publish_source_aliases_swaps_filtered_aliases_atomically(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.publish_source_aliases("aic_hcm2026_text_v1_test")

        actions = client.indices.alias_updates[0]["body"]["actions"]
        self.assertEqual(len(actions), 10)
        remove_actions = [action for action in actions if "remove" in action]
        add_actions = [action for action in actions if "add" in action]
        self.assertEqual(len(remove_actions), 5)
        self.assertEqual(len(add_actions), 5)
        self.assertTrue(
            all(
                action["remove"]["index"] == "*"
                for action in remove_actions
            )
        )
        alias_names = {action["add"]["alias"] for action in add_actions}
        self.assertEqual(alias_names, set(DEFAULT_SOURCE_ALIASES.values()))
        filters = {
            action["add"]["alias"]: action["add"]["filter"]["term"]["source_type"]
            for action in add_actions
        }
        self.assertEqual(filters[DEFAULT_SOURCE_ALIASES["ocr"]], "ocr")
        self.assertEqual(filters[DEFAULT_SOURCE_ALIASES["caption"]], "caption")

    def test_index_documents_handles_none_elements_in_input_list(self) -> None:
        captured_actions = []

        def fake_bulk(client, actions, **kwargs):
            captured_actions.extend(list(actions))
            return (1, [])

        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=fake_bulk,
        )
        document = make_document()

        summary = manager.index_documents([None, document, None], index_name="test-index")  # type: ignore[list-item]

        self.assertEqual(summary, {"indexed": 1, "failed": 0})
        self.assertEqual(len(captured_actions), 1)

    def test_index_documents_splits_large_batch_into_chunks(self) -> None:
        bulk_batches = []

        def fake_bulk(client, actions, **kwargs):
            actions_list = list(actions)
            bulk_batches.append(actions_list)
            return (len(actions_list), [])

        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=fake_bulk,
        )
        docs = [
            make_document(doc_id=f"ocr_frame:{i}:v1", entity_id=f"ent_{i}")
            for i in range(5)
        ]

        summary = manager.index_documents(docs, index_name="test-index", chunk_size=2)

        self.assertEqual(summary, {"indexed": 5, "failed": 0})
        self.assertEqual(len(bulk_batches), 3)  # 2 + 2 + 1 = 5 docs in 3 chunks
        self.assertEqual(len(bulk_batches[0]), 2)
        self.assertEqual(len(bulk_batches[1]), 2)
        self.assertEqual(len(bulk_batches[2]), 1)

    def test_index_documents_requires_explicit_physical_index_name(self) -> None:
        bulk_calls = []
        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=lambda *args, **kwargs: bulk_calls.append((args, kwargs)),
        )

        with self.assertRaisesRegex(ValueError, "index_name"):
            manager.index_documents([make_document()])

        self.assertEqual(bulk_calls, [])

    def test_index_documents_deduplicates_duplicate_doc_ids_in_batch(self) -> None:
        captured_actions = []

        def fake_bulk(client, actions, **kwargs):
            captured_actions.extend(list(actions))
            return (1, [])

        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=fake_bulk,
        )
        document = make_document()

        summary = manager.index_documents([document, document], index_name="test-index")
        self.assertEqual(summary, {"indexed": 1, "failed": 0})
        self.assertEqual(len(captured_actions), 1)

    def test_index_documents_uses_doc_id_as_bulk_id_and_reports_success(self) -> None:
        captured_actions = []

        def fake_bulk(client, actions, **kwargs):
            captured_actions.extend(list(actions))
            return (1, [])

        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=fake_bulk,
        )
        document = make_document()

        summary = manager.index_documents([document], index_name="test-index")

        self.assertEqual(summary, {"indexed": 1, "failed": 0})
        self.assertEqual(captured_actions[0]["_id"], document.doc_id)
        self.assertEqual(captured_actions[0]["_source"]["entity_id"], "L21_V001_001")

    def test_index_documents_reports_partial_failures(self) -> None:
        def fake_bulk(client, actions, **kwargs):
            list(actions)
            return (1, [{"index": {"error": "failed"}}])

        manager = ElasticsearchManager(
            client=FakeElasticsearchClient(),
            bulk_helper=fake_bulk,
        )

        summary = manager.index_documents([make_document()], index_name="test-index")

        self.assertEqual(summary, {"indexed": 1, "failed": 1})

    def test_search_routes_empty_source_types_to_all_aliases(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.search(TextSearchQuery(query_text="le trao giai"))

        self.assertEqual(
            set(client.search_calls[0]["index"]),
            set(DEFAULT_SOURCE_ALIASES.values()),
        )

    def test_search_routes_selected_source_types_to_selected_aliases(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.search(
            TextSearchQuery(query_text="le trao giai", source_types=("ocr",))
        )

        self.assertEqual(client.search_calls[0]["index"], [DEFAULT_SOURCE_ALIASES["ocr"]])

    def test_search_query_contains_filters_temporal_logic_and_spatial_region(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.search(
            TextSearchQuery(
                query_text="AIC",
                source_types=("ocr",),
                video_ids=("L21_V001",),
                language="vi",
                start_ms=0,
                end_ms=1_000,
                ocr_region="top",
            )
        )

        query = client.search_calls[0]["body"]["query"]["bool"]
        filters = query["filter"]
        self.assertIn({"terms": {"video_id": ["L21_V001"]}}, filters)
        self.assertIn({"term": {"language": "vi"}}, filters)
        temporal_filter = next(item for item in filters if "bool" in item)
        temporal_should = temporal_filter["bool"]["should"]
        self.assertIn(
            {
                "bool": {
                    "filter": [
                        {"range": {"start_ms": {"lt": 1_000}}},
                        {"range": {"end_ms": {"gt": 0}}},
                    ]
                }
            },
            temporal_should,
        )
        self.assertIn(
            {
                "bool": {
                    "filter": [
                        {"range": {"timestamp_ms": {"gte": 0}}},
                        {"range": {"timestamp_ms": {"lt": 1_000}}},
                    ]
                }
            },
            temporal_should,
        )
        self.assertTrue(any("nested" in item for item in filters))

    def test_search_flags_disable_fuzzy_and_highlight(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        manager.search(
            TextSearchQuery(
                query_text="AIC",
                use_fuzzy=False,
                use_highlight=False,
            )
        )

        body = client.search_calls[0]["body"]
        should = body["query"]["bool"]["should"]
        self.assertNotIn("highlight", body)
        self.assertFalse(any("fuzziness" in str(clause) for clause in should))

    def test_search_parses_hits_and_optional_highlights(self) -> None:
        response = {
            "hits": {
                "hits": [
                    {
                        "_id": "ocr_frame:L21_V001_001:v1",
                        "_score": 2.5,
                        "_source": {
                            "doc_id": "ocr_frame:L21_V001_001:v1",
                            "source_type": "ocr",
                            "video_id": "L21_V001",
                            "entity_id": "L21_V001_001",
                            "content": "Lễ trao giải",
                            "frame_id": "L21_V001_001",
                            "timestamp_ms": 0,
                        },
                        "highlight": {"content": ["<em>Lễ trao giải</em>"]},
                    }
                ]
            }
        }
        manager = ElasticsearchManager(client=FakeElasticsearchClient(search_response=response))

        hits = manager.search(TextSearchQuery(query_text="le trao giai"))

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].entity_id, "L21_V001_001")
        self.assertEqual(hits[0].highlights, ("<em>Lễ trao giải</em>",))

    def test_search_query_all_ocr_spatial_regions(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        for region in ("top", "header", "bottom", "footer", "left", "right", "center"):
            client.search_calls.clear()
            manager.search(TextSearchQuery(query_text="test", ocr_region=region))
            filters = client.search_calls[0]["body"]["query"]["bool"]["filter"]
            nested_filter = next(item for item in filters if "nested" in item)
            self.assertEqual(nested_filter["nested"]["path"], "regions")

    def test_search_query_start_only_and_end_only_temporal_filters(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        # Start only
        manager.search(TextSearchQuery(query_text="test", start_ms=1000))
        filters_start = client.search_calls[0]["body"]["query"]["bool"]["filter"]
        temporal_should_start = next(item for item in filters_start if "bool" in item)["bool"]["should"]
        self.assertIn({"bool": {"filter": [{"range": {"end_ms": {"gt": 1000}}}]}}, temporal_should_start)

        # End only
        client.search_calls.clear()
        manager.search(TextSearchQuery(query_text="test", end_ms=5000))
        filters_end = client.search_calls[0]["body"]["query"]["bool"]["filter"]
        temporal_should_end = next(item for item in filters_end if "bool" in item)["bool"]["should"]
        self.assertIn({"bool": {"filter": [{"range": {"start_ms": {"lt": 5000}}}]}}, temporal_should_end)

    def test_health_check_returns_info_and_aliases(self) -> None:
        client = FakeElasticsearchClient()
        manager = ElasticsearchManager(client=client)

        result_no_index = manager.health_check()
        self.assertTrue(client.info_called)
        self.assertIsNone(result_no_index["aliases"])

        result_with_index = manager.health_check(index_name="test-index")
        self.assertIsNotNone(result_with_index["aliases"])

    def test_search_with_unconfigured_source_type_raises_value_error(self) -> None:
        manager = ElasticsearchManager(client=FakeElasticsearchClient(), source_aliases={})
        query = TextSearchQuery(query_text="test", source_types=("ocr",))
        with self.assertRaisesRegex(ValueError, "No alias configured for source_type=ocr"):
            manager.search(query)

    def test_demo_module_import_has_no_live_connection_side_effect(self) -> None:
        module = importlib.import_module("BackEnd.app.database.demo_elasticsearch")

        self.assertTrue(hasattr(module, "build_sample_documents"))


def is_elasticsearch_reachable() -> bool:
    url = os.getenv("ELASTICSEARCH_URL")
    if not url or os.getenv("ELASTICSEARCH_ALLOW_TEST_INDEX_DELETE") != "true":
        return False
    try:
        import requests
        return requests.get(url, timeout=1).status_code == 200
    except Exception:
        return False


@unittest.skipUnless(
    is_elasticsearch_reachable(),
    "Active Elasticsearch server on ELASTICSEARCH_URL and ELASTICSEARCH_ALLOW_TEST_INDEX_DELETE=true are required.",
)
class ElasticsearchLiveIntegrationTests(unittest.TestCase):
    """Optional live checks using only namespaced test indexes and aliases."""

    def test_live_index_alias_index_and_search_flow(self) -> None:
        suffix = uuid4().hex[:12]
        index_name = f"aic_hcm2026_text_v1_test_{suffix}"
        source_aliases = {
            "video_metadata": f"aic_hcm2026_text_test_metadata_{suffix}",
            "ocr": f"aic_hcm2026_text_test_ocr_{suffix}",
            "transcript": f"aic_hcm2026_text_test_transcript_{suffix}",
            "caption": f"aic_hcm2026_text_test_caption_{suffix}",
        }
        manager = ElasticsearchManager(source_aliases=source_aliases)

        try:
            manager.create_index(index_name)
            manager.publish_source_aliases(index_name)
            manager.index_documents(
                build_sample_documents(),
                index_name=index_name,
                refresh=True,
            )

            ocr_hits = manager.search(
                TextSearchQuery(query_text="trao giải", source_types=("ocr",))
            )
            mixed_hits = manager.search(TextSearchQuery(query_text="trao giải"))

            self.assertTrue(ocr_hits)
            self.assertTrue(all(hit.source_type == "ocr" for hit in ocr_hits))
            self.assertTrue(
                {hit.source_type for hit in mixed_hits} & {"ocr", "transcript"}
            )
        finally:
            if not index_name.startswith("aic_hcm2026_text_v1_test_"):
                raise AssertionError(f"Unsafe test index cleanup target: {index_name}")
            manager.client.indices.delete(index=index_name, ignore_unavailable=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
