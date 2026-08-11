"""Text search service layer managing index sync and querying for AIC video retrieval."""

from __future__ import annotations

from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from BackEnd.CONFIG import ELASTICSEARCH_BULK_BATCH_SIZE, ELASTICSEARCH_STREAM_BATCH_SIZE
from BackEnd.app.contracts.search import TextIndexDocument, TextSearchHit, TextSearchQuery
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder
from BackEnd.app.database.models import Caption, Frame, ObjectDetection, TranscriptSegment, Video


class TextSearchService:
    """High-level service interface for text retrieval and PostgreSQL index sync."""

    def __init__(
        self,
        manager: ElasticsearchManager | None = None,
        builder: ElasticsearchDocumentBuilder | None = None,
        *,
        elasticsearch_url: str | None = None,
    ) -> None:
        self.manager = manager or ElasticsearchManager(elasticsearch_url=elasticsearch_url)
        self.builder = builder or ElasticsearchDocumentBuilder()

    def search(self, query: TextSearchQuery) -> list[TextSearchHit]:
        """Execute a text search query against Elasticsearch source aliases."""

        return self.manager.search(query)

    def sync_from_postgres(
        self,
        session: Session,
        *,
        index_name: str,
        index_build_id: str = "build-auto",
        batch_size: int = ELASTICSEARCH_STREAM_BATCH_SIZE,
        publish_aliases: bool = True,
        recreate_index: bool = False,
        bigdata: bool = True,
    ) -> dict[str, int]:
        """Build and index documents from PostgreSQL ORM records using paginated batching.

        Args:
            session: Active SQLAlchemy session.
            index_name: Physical Lucene index name.
            index_build_id: Build tracking identifier string.
            batch_size: Bulk batch size for streaming documents.
            publish_aliases: Atomic alias swap after indexing completes.
            recreate_index: If True, deletes existing index before building.
            bigdata: Applies BigData optimized index settings (shards, compression, refresh).
        """

        import logging as _log

        _logger = _log.getLogger(__name__)

        if recreate_index:
            if self.manager.client.indices.exists(index=index_name):
                _logger.info("Deleting existing index '%s' for clean recreate.", index_name)
                self.manager.client.indices.delete(index=index_name)

        self.manager.ensure_index_exists(index_name, bigdata=bigdata)

        total_indexed = 0
        total_failed = 0
        docs_since_last_log = 0
        counts: dict[str, int] = {
            "video_metadata_docs": 0,
            "ocr_docs": 0,
            "transcript_docs": 0,
            "caption_docs": 0,
            "object_docs": 0,
        }

        current_batch: list[TextIndexDocument] = []

        def _index_chunk(chunk_docs: list[TextIndexDocument]) -> None:
            nonlocal total_indexed, total_failed, docs_since_last_log
            if not chunk_docs:
                return
            res = self.manager.index_documents(
                chunk_docs,
                index_name=index_name,
                refresh=False,
                chunk_size=batch_size,
            )
            total_indexed += res.get("indexed", 0)
            total_failed += res.get("failed", 0)
            docs_since_last_log += len(chunk_docs)
            if docs_since_last_log >= 5000:
                _logger.info(
                    "Progress: %d docs indexed (%d failed) so far.",
                    total_indexed,
                    total_failed,
                )
                docs_since_last_log = 0

        # 1. Video metadata (lightweight, no eager loading needed)
        for video in session.scalars(select(Video)).yield_per(batch_size):
            doc = self.builder.build_video_metadata_document(
                video,
                index_build_id=index_build_id,
            )
            if doc:
                current_batch.append(doc)
                counts["video_metadata_docs"] += 1
                if len(current_batch) >= batch_size:
                    _index_chunk(current_batch)
                    current_batch = []
        session.expunge_all()

        # 2. Keyframes with OCR records — Keyset/Cursor pagination for frames WITH OCR records
        last_frame_id = ""
        while True:
            frame_query = (
                select(Frame)
                .options(selectinload(Frame.ocr_records))
                .where(Frame.frame_id > last_frame_id)
                .where(Frame.ocr_records.any())
                .order_by(Frame.frame_id)
                .limit(batch_size)
            )
            frames = list(session.scalars(frame_query).all())
            if not frames:
                break
            for frame in frames:
                if frame.ocr_records:
                    doc = self.builder.build_ocr_document(
                        frame,
                        list(frame.ocr_records),
                        index_build_id=index_build_id,
                    )
                    if doc:
                        current_batch.append(doc)
                        counts["ocr_docs"] += 1
                        if len(current_batch) >= batch_size:
                            _index_chunk(current_batch)
                            current_batch = []
            last_frame_id = frames[-1].frame_id
            session.expunge_all()

        # 3. Transcript segments (lightweight, no eager loading needed)
        for segment in session.scalars(select(TranscriptSegment)).yield_per(batch_size):
            doc = self.builder.build_transcript_document(
                segment,
                index_build_id=index_build_id,
            )
            if doc:
                current_batch.append(doc)
                counts["transcript_docs"] += 1
                if len(current_batch) >= batch_size:
                    _index_chunk(current_batch)
                    current_batch = []
        session.expunge_all()

        # 4. Captions — Keyset/Cursor pagination on caption_id (O(1) lookup per batch)
        last_caption_id = 0
        while True:
            caption_query = (
                select(Caption)
                .options(
                    joinedload(Caption.frame),
                    joinedload(Caption.shot),
                    joinedload(Caption.clip),
                )
                .where(Caption.caption_id > last_caption_id)
                .order_by(Caption.caption_id)
                .limit(batch_size)
            )
            captions = list(session.scalars(caption_query.execution_options(uniquify=True)).all())
            if not captions:
                break
            for caption in captions:
                try:
                    doc = self.builder.build_caption_document(
                        caption,
                        index_build_id=index_build_id,
                    )
                    if doc:
                        current_batch.append(doc)
                        counts["caption_docs"] += 1
                        if len(current_batch) >= batch_size:
                            _index_chunk(current_batch)
                            current_batch = []
                except Exception:
                    continue
            last_caption_id = captions[-1].caption_id
            session.expunge_all()

        # 5. Object detections — Keyset/Cursor pagination on frame_id for frames WITH detections
        last_frame_id = ""
        while True:
            frame_obj_query = (
                select(Frame)
                .options(
                    selectinload(Frame.object_detections).selectinload(
                        ObjectDetection.object_class
                    )
                )
                .where(Frame.frame_id > last_frame_id)
                .where(Frame.object_detections.any())
                .order_by(Frame.frame_id)
                .limit(batch_size)
            )
            frames = list(session.scalars(frame_obj_query).all())
            if not frames:
                break
            for frame in frames:
                if getattr(frame, "object_detections", None):
                    doc = self.builder.build_object_document(
                        frame,
                        list(frame.object_detections),
                        index_build_id=index_build_id,
                    )
                    if doc:
                        current_batch.append(doc)
                        counts["object_docs"] += 1
                        if len(current_batch) >= batch_size:
                            _index_chunk(current_batch)
                            current_batch = []
            last_frame_id = frames[-1].frame_id
            session.expunge_all()

        # Flush remaining docs and refresh Lucene index
        if current_batch:
            _index_chunk(current_batch)
            current_batch = []

        self.manager.refresh_index(index_name)

        if publish_aliases:
            self.manager.publish_source_aliases(index_name)

        _logger.info(
            "Sync complete: %d indexed, %d failed.", total_indexed, total_failed
        )

        return {
            "indexed": total_indexed,
            "failed": total_failed,

            "total_indexed": total_indexed,
            "video_metadata_docs": counts["video_metadata_docs"],
            "ocr_docs": counts["ocr_docs"],
            "transcript_docs": counts["transcript_docs"],
            "caption_docs": counts["caption_docs"],
            "object_docs": counts["object_docs"],
        }


    def health_check(self, index_name: str | None = None) -> dict[str, Any]:
        """Check Elasticsearch connectivity and optional index alias state."""

        return self.manager.health_check(index_name=index_name)
