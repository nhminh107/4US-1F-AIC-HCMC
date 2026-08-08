"""Integration tests parsing and searching real dataset metadata from data/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from BackEnd.app.contracts.search import TextIndexDocument, TextSearchQuery
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEDIA_INFO_DIR = PROJECT_ROOT / "data/media-info-aic25-b1/media-info"


class RealDataElasticsearchDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = ElasticsearchDocumentBuilder()

    def test_parse_real_video_metadata_json_files(self) -> None:
        """Parse all real video metadata JSON files from data/ into TextIndexDocument."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest(f"Media info directory not found at: {MEDIA_INFO_DIR}")

        json_paths = sorted(MEDIA_INFO_DIR.glob("*.json"))
        self.assertGreater(len(json_paths), 0, "No JSON metadata files found.")

        parsed_documents: list[TextIndexDocument] = []
        for json_path in json_paths[:50]:  # Test sample batch of 50 real JSON files
            with json_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)

            video_id = json_path.stem
            video_obj = SimpleNamespace(
                video_id=video_id,
                title=raw_data.get("title"),
                description=raw_data.get("description"),
                keywords=raw_data.get("keywords"),
                author=raw_data.get("author"),
            )

            doc = self.builder.build_video_metadata_document(
                video_obj,
                index_build_id="real-data-test-build",
            )

            self.assertIsNotNone(doc)
            self.assertEqual(doc.video_id, video_id)
            self.assertEqual(doc.source_type, "video_metadata")
            self.assertTrue(len(doc.content) > 0)
            parsed_documents.append(doc)

        self.assertEqual(len(parsed_documents), 50)


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
class RealDataElasticsearchLiveIntegrationTests(unittest.TestCase):
    """Live search integration test using real video metadata against an active Elasticsearch instance."""

    def test_live_search_on_real_video_metadata(self) -> None:
        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest(f"Media info directory not found at: {MEDIA_INFO_DIR}")

        builder = ElasticsearchDocumentBuilder()
        json_paths = sorted(MEDIA_INFO_DIR.glob("*.json"))[:30]

        documents: list[TextIndexDocument] = []
        for json_path in json_paths:
            with json_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)

            video_id = json_path.stem
            video_obj = SimpleNamespace(
                video_id=video_id,
                title=raw_data.get("title"),
                description=raw_data.get("description"),
                keywords=raw_data.get("keywords"),
                author=raw_data.get("author"),
            )
            doc = builder.build_video_metadata_document(
                video_obj,
                index_build_id="live-real-data-build",
            )
            if doc:
                documents.append(doc)

        suffix = uuid4().hex[:12]
        index_name = f"aic_hcm2026_text_v1_realdata_test_{suffix}"
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
            manager.index_documents(documents, index_name=index_name, refresh=True)

            # Test search with real Vietnamese queries
            hits = manager.search(
                TextSearchQuery(query_text="60 giây sáng", source_types=("video_metadata",))
            )
            self.assertTrue(hits)
            self.assertIn("L21_V001", [hit.video_id for hit in hits])

            hits_unaccented = manager.search(
                TextSearchQuery(query_text="htv tin tuc", source_types=("video_metadata",))
            )
            self.assertTrue(hits_unaccented)
        finally:
            if index_name.startswith("aic_hcm2026_text_v1_realdata_test_"):
                manager.client.indices.delete(index=index_name, ignore_unavailable=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
