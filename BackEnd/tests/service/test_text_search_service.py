"""Unit tests for the TextSearchService layer."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from BackEnd.app.contracts.search import TextIndexDocument, TextSearchQuery
from BackEnd.app.service.text_search_service import TextSearchService


class FakeManager:
    def __init__(self) -> None:
        self.search_queries: list[TextSearchQuery] = []
        self.indexed_docs: list[TextIndexDocument] = []
        self.published_indices: list[str] = []

    def search(self, query: TextSearchQuery) -> list:
        self.search_queries.append(query)
        return []

    def index_documents(self, documents: list[TextIndexDocument], *, index_name: str, **kwargs) -> dict[str, int]:
        self.indexed_docs.extend(documents)
        return {"indexed": len(documents), "failed": 0}

    def publish_source_aliases(self, index_name: str) -> None:
        self.published_indices.append(index_name)

    def health_check(self, index_name: str | None = None) -> dict:
        return {"status": "ok", "index": index_name}


class TextSearchServiceTests(unittest.TestCase):
    def test_service_search_delegates_to_manager(self) -> None:
        manager = FakeManager()
        service = TextSearchService(manager=manager)  # type: ignore[arg-type]

        query = TextSearchQuery(query_text="le trao giai")
        hits = service.search(query)

        self.assertEqual(hits, [])
        self.assertEqual(len(manager.search_queries), 1)
        self.assertEqual(manager.search_queries[0].query_text, "le trao giai")

    def test_service_sync_from_postgres_uses_eager_loading_and_indexes(self) -> None:
        manager = FakeManager()
        service = TextSearchService(manager=manager)  # type: ignore[arg-type]

        fake_session = MagicMock()
        fake_session.scalars.return_value.all.return_value = []

        summary = service.sync_from_postgres(
            session=fake_session,
            index_name="aic_hcm2026_test",
            batch_size=100,
        )

        self.assertEqual(summary, {"indexed": 0, "failed": 0})
        self.assertIn("aic_hcm2026_test", manager.published_indices)
        self.assertEqual(fake_session.scalars.call_count, 4)

    def test_service_health_check_delegates_to_manager(self) -> None:
        manager = FakeManager()
        service = TextSearchService(manager=manager)  # type: ignore[arg-type]

        res = service.health_check(index_name="test-index")
        self.assertEqual(res, {"status": "ok", "index": "test-index"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
