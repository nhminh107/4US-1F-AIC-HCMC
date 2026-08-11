"""Production pipeline stage: Sync PostgreSQL database records into Elasticsearch.

Official pipeline module: BackEnd/app/pipeline/sync_elasticsearch.py

Streams video metadata, keyframe OCR, transcript segments, AI captions, and
object detections from PostgreSQL into Elasticsearch physical indices, then
atomically swaps source aliases for zero-downtime search deployment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import sys
import time
from typing import Any

from BackEnd.CONFIG import (
    ELASTICSEARCH_STREAM_BATCH_SIZE,
)
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.service.text_search_service import TextSearchService

logger = logging.getLogger(__name__)


def run_elasticsearch_sync(
    *,
    index_name: str = "aic_hcm2026_text_v1",
    batch_size: int = ELASTICSEARCH_STREAM_BATCH_SIZE,
    build_id: str | None = None,
    recreate_index: bool = False,
    bigdata: bool = True,
    publish_aliases: bool = True,
    elasticsearch_url: str | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Execute official production sync from PostgreSQL to Elasticsearch.

    Returns a metrics summary dictionary including execution timing and
    throughput (documents per second).
    """

    start_time = time.perf_counter()
    auto_build_id = build_id or f"build-{datetime.now(timezone.utc).strftime('%Y%m%m-%H%M%S')}"

    print("=" * 60)
    print("🚀 PROD PIPELINE: ĐỒNG BỘ POSTGRESQL SANG ELASTICSEARCH")
    print("=" * 60)
    print(f"📦 Physical Index : {index_name}")
    print(f"🏗️  Build ID       : {auto_build_id}")
    print(f"⚡ Batch Size     : {batch_size}")
    print(f"📊 BigData Mode   : {bigdata}")
    print(f"🔄 Recreate Index : {recreate_index}")
    print(f"🔗 Swap Aliases   : {publish_aliases}")
    print("=" * 60)

    manager = ElasticsearchManager(elasticsearch_url=elasticsearch_url)

    # 1. Health check ES connectivity
    health = manager.health_check()
    if not health.get("info"):
        raise RuntimeError("Elasticsearch connection failed. Ensure Elasticsearch is running!")

    service = TextSearchService(manager=manager)
    db_mgr = PostgreManager(database_url=database_url)

    print("📦 Bắt đầu streaming nạp dữ liệu từ PostgreSQL vào Elasticsearch...")

    try:
        with db_mgr.session_factory() as session:
            sync_res = service.sync_from_postgres(
                session=session,
                index_name=index_name,
                index_build_id=auto_build_id,
                batch_size=batch_size,
                publish_aliases=publish_aliases,
                recreate_index=recreate_index,
                bigdata=bigdata,
            )

        # 2. Finalize Lucene segments (re-enable refresh, topology replica check, force merge)
        print("🔧 Đang finalize index (re-enable refresh, force merge segments)...")
        manager.finalize_bulk_index(index_name)

    finally:
        db_mgr.engine.dispose()

    elapsed_seconds = round(time.perf_counter() - start_time, 2)
    total_indexed = sync_res.get("total_indexed", 0)
    throughput = round(total_indexed / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0

    print("\n🎉 HOÀN THÀNH ĐỒNG BỘ DỮ LIỆU SẢN XUẤT (PRODUCTION SYNC COMPLETE)!")
    print("=" * 60)
    print(f"📊 Tổng số bản ghi đã nạp  : {total_indexed:,}")
    print(f"🎬 Video Metadata Documents: {sync_res.get('video_metadata_docs', 0):,}")
    print(f"📝 OCR Frame Documents     : {sync_res.get('ocr_docs', 0):,}")
    print(f"💬 Transcript Documents    : {sync_res.get('transcript_docs', 0):,}")
    print(f"🖼️ Caption Documents       : {sync_res.get('caption_docs', 0):,}")
    print(f"🎯 Object Detections       : {sync_res.get('object_docs', 0):,}")
    print("-" * 60)
    print(f"⏱️  Thời gian thực thi     : {elapsed_seconds} giây")
    print(f"🚀 Tốc độ nạp (Throughput)  : {throughput:,} docs/giây")
    print("=" * 60)

    return {
        "status": "success",
        "index_name": index_name,
        "build_id": auto_build_id,
        "elapsed_seconds": elapsed_seconds,
        "throughput_docs_per_sec": throughput,
        **sync_res,
    }


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Production Pipeline Stage: PostgreSQL -> Elasticsearch Sync",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="aic_hcm2026_text_v1",
        help="Physical Lucene index name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ELASTICSEARCH_STREAM_BATCH_SIZE,
        help="Bulk batch size for streaming records",
    )
    parser.add_argument(
        "--build-id",
        type=str,
        default=None,
        help="Build tracking identifier string",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete existing physical index and rebuild from scratch",
    )
    parser.add_argument(
        "--no-bigdata",
        action="store_true",
        help="Disable BigData optimized settings (3 shards, best_compression, refresh=off)",
    )
    parser.add_argument(
        "--no-publish-aliases",
        action="store_true",
        help="Do not publish/swap active source aliases after indexing",
    )
    parser.add_argument(
        "--elasticsearch-url",
        type=str,
        default=None,
        help="Elasticsearch server URL override",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="PostgreSQL database connection URL override",
    )
    return parser.parse_args(args)


def main() -> None:
    # Ensure Windows stdout handles UTF-8 correctly
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parsed = _parse_args()
    try:
        run_elasticsearch_sync(
            index_name=parsed.index_name,
            batch_size=parsed.batch_size,
            build_id=parsed.build_id,
            recreate_index=parsed.recreate_index,
            bigdata=not parsed.no_bigdata,
            publish_aliases=not parsed.no_publish_aliases,
            elasticsearch_url=parsed.elasticsearch_url,
            database_url=parsed.database_url,
        )
    except Exception as err:
        print(f"❌ Production Sync Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
