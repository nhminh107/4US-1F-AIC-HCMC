"""Text search service layer managing index sync and querying for AIC video retrieval."""

from __future__ import annotations

from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from BackEnd.CONFIG import ELASTICSEARCH_BULK_BATCH_SIZE
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
        batch_size: int = ELASTICSEARCH_BULK_BATCH_SIZE,
        publish_aliases: bool = True,
    ) -> dict[str, int]:
        """Build and index documents from PostgreSQL ORM records using eager loading.

        Prevents N+1 queries by pre-fetching relationships (OCR records, captions).
        Returns total count of indexed and failed documents.
        """

        documents: list[TextIndexDocument] = []

        # 1. Video metadata
        videos = session.scalars(select(Video)).all()
        for video in videos:
            doc = self.builder.build_video_metadata_document(
                video,
                index_build_id=index_build_id,
            )
            if doc:
                documents.append(doc)

        # 2. Keyframes with OCR records (eager load ocr_records to avoid N+1)
        frames = session.scalars(
            select(Frame).options(selectinload(Frame.ocr_records))
        ).all()
        for frame in frames:
            if frame.ocr_records:
                doc = self.builder.build_ocr_document(
                    frame,
                    list(frame.ocr_records),
                    index_build_id=index_build_id,
                )
                if doc:
                    documents.append(doc)

        # 3. Transcript segments
        transcripts = session.scalars(select(TranscriptSegment)).all()
        for segment in transcripts:
            doc = self.builder.build_transcript_document(
                segment,
                index_build_id=index_build_id,
            )
            if doc:
                documents.append(doc)

        # 4. Captions (eager load frame, shot, clip to resolve video_id without N+1)
        captions = session.scalars(
            select(Caption).options(
                joinedload(Caption.frame),
                joinedload(Caption.shot),
                joinedload(Caption.clip),
            )
        ).all()
        for caption in captions:
            try:
                doc = self.builder.build_caption_document(
                    caption,
                    index_build_id=index_build_id,
                )
                if doc:
                    documents.append(doc)
            except ValueError:
                continue

        # Create physical index if not created yet, then bulk index documents
        summary = self.manager.index_documents(
            documents,
            index_name=index_name,
            refresh=True,
            chunk_size=batch_size,
        )

        if publish_aliases:
            self.manager.publish_source_aliases(index_name)

        return summary

    def health_check(self, index_name: str | None = None) -> dict[str, Any]:
        """Check Elasticsearch connectivity and optional index alias state."""

        return self.manager.health_check(index_name=index_name)
