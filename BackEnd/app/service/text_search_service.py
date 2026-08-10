"""Text search service layer managing index sync and querying for AIC video retrieval."""

from __future__ import annotations

from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from BackEnd.CONFIG import ELASTICSEARCH_BULK_BATCH_SIZE, ELASTICSEARCH_STREAM_BATCH_SIZE
from BackEnd.app.contracts.search import TextIndexDocument, TextSearchHit, TextSearchQuery
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder
from BackEnd.app.database.models import Caption, Frame, TranscriptSegment, Video


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
    ) -> dict[str, int]:
        """Build and index documents from PostgreSQL ORM records using streaming batching.

        Prevents memory exhaustion by processing records in chunks, flushing each chunk to
        Elasticsearch with refresh=False, and clearing the SQLAlchemy session.
        Returns total count of indexed and failed documents.
        """

        total_indexed = 0
        total_failed = 0
        current_batch: list[TextIndexDocument] = []

        def _index_chunk(chunk_docs: list[TextIndexDocument]) -> None:
            nonlocal total_indexed, total_failed
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

        # 1. Video metadata
        for video in session.scalars(select(Video)).yield_per(batch_size):
            doc = self.builder.build_video_metadata_document(
                video,
                index_build_id=index_build_id,
            )
            if doc:
                current_batch.append(doc)
                if len(current_batch) >= batch_size:
                    _index_chunk(current_batch)
                    current_batch = []
                    session.clear()

        # 2. Keyframes with OCR records
        frame_query = select(Frame).options(selectinload(Frame.ocr_records))
        for frame in session.scalars(frame_query).yield_per(batch_size):
            if frame.ocr_records:
                doc = self.builder.build_ocr_document(
                    frame,
                    list(frame.ocr_records),
                    index_build_id=index_build_id,
                )
                if doc:
                    current_batch.append(doc)
                    if len(current_batch) >= batch_size:
                        _index_chunk(current_batch)
                        current_batch = []
                        session.clear()

        # 3. Transcript segments
        for segment in session.scalars(select(TranscriptSegment)).yield_per(batch_size):
            doc = self.builder.build_transcript_document(
                segment,
                index_build_id=index_build_id,
            )
            if doc:
                current_batch.append(doc)
                if len(current_batch) >= batch_size:
                    _index_chunk(current_batch)
                    current_batch = []
                    session.clear()

        # 4. Captions
        caption_query = select(Caption).options(
            joinedload(Caption.frame),
            joinedload(Caption.shot),
            joinedload(Caption.clip),
        )
        for caption in session.scalars(caption_query).yield_per(batch_size):
            try:
                doc = self.builder.build_caption_document(
                    caption,
                    index_build_id=index_build_id,
                )
                if doc:
                    current_batch.append(doc)
                    if len(current_batch) >= batch_size:
                        _index_chunk(current_batch)
                        current_batch = []
                        session.clear()
            except Exception:
                continue

        # Flush remaining docs and refresh Lucene index
        if current_batch:
            _index_chunk(current_batch)
            current_batch = []
            session.clear()

        self.manager.refresh_index(index_name)

        if publish_aliases:
            self.manager.publish_source_aliases(index_name)

        return {"indexed": total_indexed, "failed": total_failed}

    def health_check(self, index_name: str | None = None) -> dict[str, Any]:
        """Check Elasticsearch connectivity and optional index alias state."""

        return self.manager.health_check(index_name=index_name)
