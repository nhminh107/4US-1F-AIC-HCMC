"""Pipeline stage to sync all text evidence from PostgreSQL to Elasticsearch."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys

from sqlalchemy.orm import Session

from BackEnd.CONFIG import ELASTICSEARCH_BULK_BATCH_SIZE
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.service.text_search_service import TextSearchService

logger = logging.getLogger(__name__)


def run_elasticsearch_ingestion(
    *,
    index_name: str | None = None,
    index_build_id: str | None = None,
    batch_size: int = ELASTICSEARCH_BULK_BATCH_SIZE,
    publish_aliases: bool = True,
    elasticsearch_url: str | None = None,
    db: PostgreManager | None = None,
) -> dict[str, int]:
    """Extract text data from PostgreSQL and ingest into Elasticsearch with active aliases."""

    db_manager = db or PostgreManager()
    service = TextSearchService(elasticsearch_url=elasticsearch_url)

    build_id = index_build_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    target_index = index_name or f"aic_hcm2026_text_v1_{build_id}"

    logging.info(
        "Starting text data sync from PostgreSQL to Elasticsearch index '%s' (build_id: %s)...",
        target_index,
        build_id,
    )

    with Session(db_manager.engine) as session:
        summary = service.sync_from_postgres(
            session=session,
            index_name=target_index,
            index_build_id=build_id,
            batch_size=batch_size,
            publish_aliases=publish_aliases,
        )

    logging.info(
        "Sync completed: %d documents indexed, %d failed. Active aliases published: %s",
        summary.get("indexed", 0),
        summary.get("failed", 0),
        publish_aliases,
    )

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sync text data (metadata, OCR, transcripts, captions) from PostgreSQL into Elasticsearch."
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Physical index name (defaults to 'aic_hcm2026_text_v1_<timestamp>').",
    )
    parser.add_argument(
        "--build-id",
        type=str,
        default=None,
        help="Index build identifier (defaults to current timestamp).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ELASTICSEARCH_BULK_BATCH_SIZE,
        help=f"Bulk indexing chunk size (default: {ELASTICSEARCH_BULK_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-publish-aliases",
        action="store_true",
        help="Do not update active source aliases after indexing.",
    )
    args = parser.parse_args()

    try:
        summary = run_elasticsearch_ingestion(
            index_name=args.index_name,
            index_build_id=args.build_id,
            batch_size=args.batch_size,
            publish_aliases=not args.no_publish_aliases,
        )
        print(f"\n[+] Elasticsearch Ingestion Summary: {summary}")
    except Exception as e:
        logger.error("Failed to ingest text data into Elasticsearch: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
