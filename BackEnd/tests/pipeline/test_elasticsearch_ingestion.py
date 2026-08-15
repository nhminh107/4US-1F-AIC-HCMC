"""Unit tests for the elasticsearch_ingestion pipeline stage."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from BackEnd.app.pipeline.elasticsearch_ingestion import run_elasticsearch_ingestion


class ElasticsearchIngestionTests(unittest.TestCase):
    @patch("BackEnd.app.pipeline.elasticsearch_ingestion.Session")
    @patch("BackEnd.app.pipeline.elasticsearch_ingestion.TextSearchService")
    def test_run_elasticsearch_ingestion_delegates_to_service(
        self,
        mock_service_cls: MagicMock,
        mock_session_cls: MagicMock,
    ) -> None:
        mock_service = MagicMock()
        mock_service.sync_from_postgres.return_value = {"indexed": 42, "failed": 0}
        mock_service_cls.return_value = mock_service

        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        summary = run_elasticsearch_ingestion(
            index_name="aic_hcm2026_text_v1_custom_test",
            index_build_id="test-build-123",
            batch_size=250,
            publish_aliases=True,
            db=mock_db,
        )

        self.assertEqual(summary, {"indexed": 42, "failed": 0})
        mock_service.sync_from_postgres.assert_called_once_with(
            session=mock_session,
            index_name="aic_hcm2026_text_v1_custom_test",
            index_build_id="test-build-123",
            batch_size=250,
            publish_aliases=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
