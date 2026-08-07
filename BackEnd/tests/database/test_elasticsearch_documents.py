"""Unit tests for PostgreSQL ORM to Elasticsearch document builders."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
import unittest

from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder


class ElasticsearchDocumentBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = ElasticsearchDocumentBuilder()

    def test_video_metadata_document_includes_searchable_fields(self) -> None:
        video = SimpleNamespace(
            video_id="L21_V001",
            title="Lễ trao giải AIC",
            description="Một buổi lễ ngoài trời",
            keywords=["AIC", "HCMC", "AIC"],
            author="4US",
            publish_date=date(2026, 8, 6),
        )

        document = self.builder.build_video_metadata_document(
            video,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.source_type, "video_metadata")
        self.assertEqual(document.entity_id, "L21_V001")
        self.assertIn("Lễ trao giải AIC", document.content)
        self.assertEqual(document.keywords, ("AIC", "HCMC", "AIC"))

    def test_video_metadata_without_searchable_text_returns_none(self) -> None:
        video = SimpleNamespace(
            video_id="L21_V001",
            title=None,
            description=None,
            keywords=None,
        )

        document = self.builder.build_video_metadata_document(
            video,
            index_build_id="build-test",
        )

        self.assertIsNone(document)

    def test_ocr_document_sorts_rows_and_preserves_regions(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=2,
                text="giải",
                language="vi",
                x_min=0.2,
                x_max=0.4,
                y_min=0.1,
                y_max=0.2,
            ),
            SimpleNamespace(
                n=1,
                text="trao",
                language="vi",
                x_min=0.1,
                x_max=0.2,
                y_min=0.1,
                y_max=0.2,
            ),
        ]

        document = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.content, "trao giải")
        self.assertEqual(document.entity_id, "L21_V001_001")
        self.assertEqual(document.timestamp_ms, 0)
        self.assertEqual(document.regions[0]["n"], 1)
        self.assertEqual(document.regions[1]["n"], 2)

    def test_ocr_document_skips_invalid_coordinates(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="bad_box",
                language="vi",
                x_min=0.5,
                x_max=0.4,
                y_min=0.1,
                y_max=0.2,
            ),
            SimpleNamespace(
                n=2,
                text="good_box",
                language="vi",
                x_min=0.1,
                x_max=0.4,
                y_min=0.1,
                y_max=0.2,
            ),
        ]

        doc = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.content, "good_box")

    def test_ocr_with_empty_text_returns_none(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="   ",
                language="vi",
                x_min=0.1,
                x_max=0.2,
                y_min=0.1,
                y_max=0.2,
            )
        ]

        document = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNone(document)

    def test_transcript_document_preserves_time_range(self) -> None:
        segment = SimpleNamespace(
            segment_id="seg-001",
            video_id="L21_V001",
            start_ms=0,
            end_ms=3_000,
            text="Xin chào AIC",
            language="vi",
        )

        document = self.builder.build_transcript_document(
            segment,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.doc_id, "transcript:seg-001:v1")
        self.assertEqual(document.start_ms, 0)
        self.assertEqual(document.end_ms, 3_000)

    def test_transcript_with_empty_text_returns_none(self) -> None:
        segment = SimpleNamespace(
            segment_id="seg-001",
            video_id="L21_V001",
            start_ms=0,
            end_ms=3_000,
            text="",
            language="vi",
        )

        document = self.builder.build_transcript_document(
            segment,
            index_build_id="build-test",
        )

        self.assertIsNone(document)

    def test_caption_document_supports_frame_target_and_model_metadata(self) -> None:
        caption = SimpleNamespace(
            caption_id=7,
            frame_id="L21_V001_001",
            clip_id=None,
            shot_id=None,
            caption_text="Một người đang phát biểu",
            model_name="vlm-test",
            model_version="1.0",
            prompt_version="caption-v1",
        )

        document = self.builder.build_caption_document(
            caption,
            index_build_id="build-test",
            video_id="L21_V001",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.doc_id, "caption:7:v1")
        self.assertEqual(document.entity_id, "7")
        self.assertEqual(document.frame_id, "L21_V001_001")
        self.assertEqual(document.model_name, "vlm-test")
        self.assertEqual(document.prompt_version, "caption-v1")

    def test_caption_with_empty_text_returns_none(self) -> None:
        caption = SimpleNamespace(
            caption_id=7,
            frame_id="L21_V001_001",
            clip_id=None,
            shot_id=None,
            caption_text=" ",
            model_name="vlm-test",
            model_version=None,
            prompt_version=None,
        )

        document = self.builder.build_caption_document(
            caption,
            index_build_id="build-test",
            video_id="L21_V001",
        )

        self.assertIsNone(document)

    def test_builder_does_not_mutate_input_lists(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=2,
                text="second",
                language="en",
                x_min=0.2,
                x_max=0.3,
                y_min=0.1,
                y_max=0.2,
            ),
            SimpleNamespace(
                n=1,
                text="first",
                language="en",
                x_min=0.1,
                x_max=0.2,
                y_min=0.1,
                y_max=0.2,
            ),
        ]

        self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertEqual([record.n for record in ocr_records], [2, 1])

    def test_video_metadata_document_parses_comma_separated_keywords_string(self) -> None:
        video = SimpleNamespace(
            video_id="L21_V001",
            title="Lễ trao giải AIC",
            description="Một buổi lễ ngoài trời",
            keywords="AIC, HCMC, 2026",
        )

        document = self.builder.build_video_metadata_document(
            video,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.keywords, ("AIC", "HCMC", "2026"))
        self.assertIn("AIC HCMC 2026", document.content)

    def test_ocr_document_allows_zero_width_or_height_valid_boxes(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="line",
                language="vi",
                x_min=0.5,
                x_max=0.5,
                y_min=0.1,
                y_max=0.2,
            )
        ]

        document = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.content, "line")

    def test_video_metadata_document_with_empty_or_whitespace_keywords_string(self) -> None:
        video = SimpleNamespace(
            video_id="L21_V001",
            title="Title",
            description=None,
            keywords="  , ,   ",
        )

        doc = self.builder.build_video_metadata_document(
            video,
            index_build_id="build-test",
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.keywords, ())
        self.assertEqual(doc.content, "Title")

    def test_ocr_document_with_all_corrupt_boxes_returns_none(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="corrupt",
                language="vi",
                x_min=1.5,
                x_max=0.4,
                y_min=0.1,
                y_max=0.2,
            )
        ]

        doc = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNone(doc)

    def test_ocr_document_preserves_first_non_empty_language(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id=None,
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="first",
                language="",
                x_min=0.1,
                x_max=0.2,
                y_min=0.1,
                y_max=0.2,
            ),
            SimpleNamespace(
                n=2,
                text="second",
                language="en",
                x_min=0.3,
                x_max=0.4,
                y_min=0.1,
                y_max=0.2,
            ),
        ]

        doc = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.language, "en")

    def test_caption_document_resolves_video_id_from_shot_and_clip_relationships(self) -> None:
        shot_caption = SimpleNamespace(
            caption_id=10,
            frame_id=None,
            clip_id=None,
            shot_id="L21_V001_S001",
            caption_text="Shot caption",
            shot=SimpleNamespace(video_id="L21_V001"),
            model_name="vlm",
            model_version="1.0",
            prompt_version="p1",
        )

        doc = self.builder.build_caption_document(
            shot_caption,
            index_build_id="build-test",
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.video_id, "L21_V001")

    def test_caption_document_unresolved_video_id_raises_value_error(self) -> None:
        orphaned_caption = SimpleNamespace(
            caption_id=11,
            frame_id=None,
            clip_id=None,
            shot_id=None,
            caption_text="Orphaned caption",
            frame=None,
            shot=None,
            clip=None,
            model_name="vlm",
            model_version="1.0",
            prompt_version="p1",
        )

        with self.assertRaisesRegex(ValueError, "caption video_id could not be resolved"):
            self.builder.build_caption_document(
                orphaned_caption,
                index_build_id="build-test",
            )

    def test_ocr_document_with_boundary_coordinates_0_and_1(self) -> None:
        frame = SimpleNamespace(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            shot_id="L21_V001_S000",
            timestamp_ms=0,
        )
        ocr_records = [
            SimpleNamespace(
                n=1,
                text="full_frame_text",
                language="vi",
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=1.0,
            )
        ]

        document = self.builder.build_ocr_document(
            frame,
            ocr_records,
            index_build_id="build-test",
        )

        self.assertIsNotNone(document)
        self.assertEqual(document.content, "full_frame_text")
        self.assertEqual(document.regions[0]["x_min"], 0.0)
        self.assertEqual(document.regions[0]["x_max"], 1.0)

    def test_caption_document_resolves_video_id_from_clip_shot_chain(self) -> None:
        clip_caption = SimpleNamespace(
            caption_id=12,
            frame_id=None,
            clip_id="C001",
            shot_id=None,
            caption_text="Clip caption",
            clip=SimpleNamespace(
                shot=SimpleNamespace(video_id="L21_V002")
            ),
            model_name="vlm",
            model_version="1.0",
            prompt_version="p1",
        )

        doc = self.builder.build_caption_document(
            clip_caption,
            index_build_id="build-test",
        )

        self.assertIsNotNone(doc)
        self.assertEqual(doc.video_id, "L21_V002")


if __name__ == "__main__":
    unittest.main(verbosity=2)
