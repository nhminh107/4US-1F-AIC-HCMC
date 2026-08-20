from __future__ import annotations

import unittest

from BackEnd.app.contracts.online_retrieval import ObjectConstraint, ObjectRetrievalRequest
from BackEnd.app.contracts.search import TextSearchHit, TextSearchQuery
from BackEnd.app.retrieval.text_object_tools import TextRetrievalTool


class _TextServiceStub:
    def search(self, query: TextSearchQuery) -> list[TextSearchHit]:
        return [
            TextSearchHit(
                doc_id="ocr-1",
                source_type="ocr",
                score=3.0,
                video_id="L01_V001",
                entity_id="F001",
                content="Circle K",
                frame_id="F001",
                timestamp_ms=1200,
            )
        ]


class TextRetrievalToolTest(unittest.TestCase):
    def test_normalizes_text_hit_for_candidate_fusion(self) -> None:
        tool = TextRetrievalTool(service=_TextServiceStub())

        candidates = tool.search(TextSearchQuery(query_text="Circle K"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "ocr")
        self.assertEqual(candidates[0].entity_type, "frame")
        self.assertEqual(candidates[0].timestamp_ms, 1200)
        self.assertAlmostEqual(candidates[0].score, 0.75)

    def test_object_request_rejects_disabled_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            ObjectRetrievalRequest(
                objects=(ObjectConstraint("person"),),
                include_detections=False,
                include_tracks=False,
            )


if __name__ == "__main__":
    unittest.main()
