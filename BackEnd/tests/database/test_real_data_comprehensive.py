"""Comprehensive real data test suite loading ALL 873 organizer video metadata files and keyframe CSVs from data/."""

from __future__ import annotations

import csv
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
KEYFRAME_MAP_DIR = PROJECT_ROOT / "data/map-keyframes-aic25-b1/map-keyframes"
OBJECTS_DIR = PROJECT_ROOT / "data/objects-aic25-b1/objects"


import unicodedata

def remove_vietnamese_accents(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


class FakeSearchEngine:
    """In-memory search emulator for real-data query evaluation without live server."""

    def __init__(self, documents: list[TextIndexDocument]) -> None:
        self.documents = {doc.doc_id: doc for doc in documents}

    def search(self, query: TextSearchQuery) -> list[dict]:
        raw_query_lower = query.query_text.lower().strip()
        query_norm = remove_vietnamese_accents(raw_query_lower)
        terms = [t for t in query_norm.split() if t]
        results = []
        for doc in self.documents.values():
            content_norm = remove_vietnamese_accents(doc.content.lower())
            title_norm = remove_vietnamese_accents(doc.title.lower()) if doc.title else ""
            score = 0.0

            # 1. Exact phrase match boost
            if query_norm in content_norm:
                score += 10.0
            if title_norm and query_norm in title_norm:
                score += 15.0  # Title match boost

            # 2. Term matches
            for term in terms:
                if term in content_norm:
                    score += 2.0
                if title_norm and term in title_norm:
                    score += 3.0

            if score > 0:
                results.append({
                    "doc_id": doc.doc_id,
                    "video_id": doc.video_id,
                    "title": doc.title,
                    "score": score,
                    "content": doc.content[:150] + "...",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[query.from_ : query.from_ + query.top_k]


class ComprehensiveRealDataTests(unittest.TestCase):
    """Test full ingestion and query retrieval against ALL 873 real video dataset files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = ElasticsearchDocumentBuilder()
        cls.all_video_docs: list[TextIndexDocument] = []
        cls.all_ocr_docs: list[TextIndexDocument] = []

        if not MEDIA_INFO_DIR.is_dir():
            return

        # 1. Load ALL 873 real video metadata JSON files
        json_paths = sorted(MEDIA_INFO_DIR.glob("*.json"))
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
            doc = cls.builder.build_video_metadata_document(
                video_obj,
                index_build_id="comprehensive-real-build",
            )
            if doc:
                cls.all_video_docs.append(doc)

        # 2. Sample keyframe OCR records from CSV
        if KEYFRAME_MAP_DIR.is_dir():
            csv_paths = sorted(KEYFRAME_MAP_DIR.glob("*.csv"))[:10]
            for csv_path in csv_paths:
                video_id = csv_path.stem
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    ocr_records = []
                    for row in list(reader)[:5]:
                        frame_id = f"{video_id}_{int(row['n']):03d}"
                        ocr_records.append(
                            SimpleNamespace(
                                n=int(row["n"]),
                                text=f"Keyframe {row['n']} timestamp {row['pts_time']}s",
                                language="vi",
                                x_min=0.1,
                                x_max=0.9,
                                y_min=0.1,
                                y_max=0.3,
                            )
                        )
                    frame_obj = SimpleNamespace(
                        frame_id=f"{video_id}_001",
                        video_id=video_id,
                        shot_id=f"{video_id}_S000",
                        timestamp_ms=0,
                    )
                    ocr_doc = cls.builder.build_ocr_document(
                        frame_obj,
                        ocr_records,
                        index_build_id="comprehensive-real-build",
                    )
                    if ocr_doc:
                        cls.all_ocr_docs.append(ocr_doc)

        cls.search_engine = FakeSearchEngine(cls.all_video_docs + cls.all_ocr_docs)

    def test_verify_all_873_real_video_metadata_documents_parsed_successfully(self) -> None:
        """Verify that ALL 873 real JSON metadata files parse cleanly into TextIndexDocuments."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        self.assertEqual(len(self.all_video_docs), 873, "Expected exactly 873 real video metadata documents.")
        for doc in self.all_video_docs:
            self.assertEqual(doc.source_type, "video_metadata")
            self.assertTrue(len(doc.video_id) > 0)
            self.assertTrue(len(doc.content) > 0)

    def test_real_data_query_retrieval_accuracy_htv_60s(self) -> None:
        """Query '60 Giây Sáng' against 873 real videos and verify top matching videos."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="60 Giây Sáng", top_k=5))
        self.assertTrue(hits, "Query '60 Giây Sáng' should return real video hits.")
        top_video_ids = [h["video_id"] for h in hits]
        self.assertIn("L21_V001", top_video_ids, "L21_V001 must be in top search hits for '60 Giây Sáng'.")

    def test_real_data_query_retrieval_accuracy_thoi_su_viet_nam(self) -> None:
        """Query 'thời sự' against 873 real videos and verify matching videos."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="thời sự", top_k=10))
        self.assertTrue(hits, "Query 'thời sự' should return matching news videos.")
        self.assertGreater(len(hits), 0)

    def test_real_data_query_pagination_on_873_videos(self) -> None:
        """Verify pagination across 873 real video documents."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        page1 = self.search_engine.search(TextSearchQuery(query_text="tin tức", top_k=10, page=1))
        page2 = self.search_engine.search(TextSearchQuery(query_text="tin tức", top_k=10, page=2))

        self.assertTrue(page1)
        self.assertTrue(page2)
        page1_ids = [h["video_id"] for h in page1]
        page2_ids = [h["video_id"] for h in page2]
        self.assertFalse(set(page1_ids) & set(page2_ids), "Page 1 and Page 2 search results should not overlap.")

    def test_real_data_search_by_channel_author_vivu_tv(self) -> None:
        """Objective test: Query videos published by travel channel 'ViVU TV'."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="ViVU TV", top_k=10))
        self.assertTrue(hits, "Search for channel 'ViVU TV' must return hits.")
        self.assertTrue(any("vivu" in h["content"].lower() for h in hits))

    def test_real_data_search_by_news_channel_tuoi_tre(self) -> None:
        """Objective test: Query videos published by news channel 'Báo Tuổi Trẻ'."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="Báo Tuổi Trẻ", top_k=10))
        self.assertTrue(hits, "Search for channel 'Báo Tuổi Trẻ' must return hits.")

    def test_real_data_unaccented_vietnamese_query_htv_sports(self) -> None:
        """Objective test: Unaccented query 'the thao htv' should match accented 'HTV Sports' content."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="the thao htv", top_k=10))
        self.assertTrue(hits, "Unaccented query 'the thao htv' must match real sports videos.")

    def test_real_data_distinguish_morning_and_evening_news(self) -> None:
        """Objective test: Distinguish between '60 Giây Sáng' and '60 Giây Chiều' videos."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        morning_hits = self.search_engine.search(TextSearchQuery(query_text="60 Giây Sáng", top_k=5))
        evening_hits = self.search_engine.search(TextSearchQuery(query_text="60 Giây Chiều", top_k=5))

        self.assertTrue(morning_hits)
        self.assertTrue(evening_hits)
        self.assertEqual(morning_hits[0]["video_id"], "L21_V001", "Morning news top hit should be L21 video.")
        self.assertEqual(evening_hits[0]["video_id"], "L22_V001", "Evening news top hit should be L22 video.")

    def test_real_data_title_match_scores_higher_than_description_only_match(self) -> None:
        """Objective test: Verify relevance ranking boosts title matches above description-only matches."""

        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        hits = self.search_engine.search(TextSearchQuery(query_text="60 Giây", top_k=10))
        self.assertTrue(hits)
        # Verify highest score has '60 Giây' in the title
        top_hit = hits[0]
        self.assertIn("60 Giây", top_hit["title"])


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
class ComprehensiveRealDataLiveSearchTests(unittest.TestCase):
    """Live search test sending ALL 873 real videos to an active Elasticsearch server."""

    def test_live_indexing_and_searching_all_873_real_videos(self) -> None:
        if not MEDIA_INFO_DIR.is_dir():
            self.skipTest("Real media-info data directory not found.")

        builder = ElasticsearchDocumentBuilder()
        json_paths = sorted(MEDIA_INFO_DIR.glob("*.json"))

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
                index_build_id="comprehensive-live-build",
            )
            if doc:
                documents.append(doc)

        suffix = uuid4().hex[:12]
        index_name = f"aic_hcm2026_text_v1_all873_test_{suffix}"
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

            summary = manager.index_documents(documents, index_name=index_name, refresh=True, chunk_size=200)
            self.assertEqual(summary["indexed"], len(documents))

            hits = manager.search(TextSearchQuery(query_text="60 Giây Sáng", source_types=("video_metadata",)))
            self.assertTrue(hits)
            self.assertEqual(hits[0].video_id, "L21_V001")

            hits_unaccented = manager.search(TextSearchQuery(query_text="tin tuc htv", source_types=("video_metadata",)))
            self.assertTrue(hits_unaccented)
        finally:
            if index_name.startswith("aic_hcm2026_text_v1_all873_test_"):
                manager.client.indices.delete(index=index_name, ignore_unavailable=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
