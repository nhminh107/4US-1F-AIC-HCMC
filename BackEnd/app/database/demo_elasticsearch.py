"""Non-destructive usage demo for the Elasticsearch text-index adapter."""

from __future__ import annotations

from BackEnd.app.contracts.search import TextIndexDocument, TextSearchQuery
from BackEnd.app.database.elasticsearch_db import (
    DEFAULT_SOURCE_ALIASES,
    INDEX_SCHEMA_VERSION,
    ElasticsearchManager,
)

DEMO_INDEX_NAME = "aic_hcm2026_text_v1_demo"
DEMO_BUILD_ID = "text-index-demo-build"


def build_sample_documents() -> list[TextIndexDocument]:
    """Return sample documents for all supported text source types."""

    return [
        TextIndexDocument(
            doc_id="video_metadata:L21_V001:v1",
            source_type="video_metadata",
            content="Lễ trao giải AIC HCMC",
            video_id="L21_V001",
            entity_id="L21_V001",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=DEMO_BUILD_ID,
            title="Lễ trao giải AIC HCMC",
            keywords=("AIC", "HCMC"),
        ),
        TextIndexDocument(
            doc_id="ocr_frame:L21_V001_001:v1",
            source_type="ocr",
            content="Trao giải",
            video_id="L21_V001",
            entity_id="L21_V001_001",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=DEMO_BUILD_ID,
            frame_id="L21_V001_001",
            timestamp_ms=0,
            regions=(
                {
                    "n": 1,
                    "text": "Trao giải",
                    "language": "vi",
                    "x_min": 0.1,
                    "x_max": 0.4,
                    "y_min": 0.1,
                    "y_max": 0.2,
                },
            ),
        ),
        TextIndexDocument(
            doc_id="transcript:seg-001:v1",
            source_type="transcript",
            content="Xin chào mừng đến với lễ trao giải",
            video_id="L21_V001",
            entity_id="seg-001",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=DEMO_BUILD_ID,
            segment_id="seg-001",
            start_ms=0,
            end_ms=3_000,
            language="vi",
        ),
        TextIndexDocument(
            doc_id="caption:1:v1",
            source_type="caption",
            content="Một người đang phát biểu trên sân khấu",
            video_id="L21_V001",
            entity_id="1",
            index_schema_version=INDEX_SCHEMA_VERSION,
            index_build_id=DEMO_BUILD_ID,
            caption_id=1,
            frame_id="L21_V001_001",
            model_name="demo-vlm",
        ),
    ]


def run_demo(elasticsearch_url: str) -> None:
    """Create a demo index and search it. This connects only when called."""

    manager = ElasticsearchManager(elasticsearch_url=elasticsearch_url)
    manager.create_index(DEMO_INDEX_NAME)
    manager.publish_source_aliases(DEMO_INDEX_NAME)
    manager.index_documents(build_sample_documents(), index_name=DEMO_INDEX_NAME)
    hits = manager.search(TextSearchQuery(query_text="le trao giai"))
    for hit in hits:
        print(f"{hit.source_type}: {hit.entity_id} score={hit.score}")


if __name__ == "__main__":
    print("Sample aliases:", DEFAULT_SOURCE_ALIASES)
    print("Sample documents:", len(build_sample_documents()))
    print("Call run_demo(elasticsearch_url) explicitly to connect to Elasticsearch.")
