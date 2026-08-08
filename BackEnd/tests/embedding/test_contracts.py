from __future__ import annotations

import unittest

from BackEnd.app.contracts.embedding import (
    ClipRecord,
    EmbeddingRecord,
    EmbeddingStatus,
    EntityType,
)
from BackEnd.app.embedding.common.ids import deterministic_id


class EmbeddingContractTests(unittest.TestCase):
    def test_clip_record_rejects_invalid_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "end_ms"):
            ClipRecord(
                clip_id="clip-1",
                video_id="L21_V001",
                shot_id="shot-1",
                start_ms=1000,
                end_ms=1000,
                scale_type="full_shot",
                target_num_frames=16,
                sampling_strategy="uniform_midpoint",
                sampling_version="clip-sampling@1.0.0",
                clip_builder_version="clip-builder@1.0.0",
            )

    def test_embedding_record_accepts_success_clip_metadata(self) -> None:
        record = EmbeddingRecord(
            embedding_id="embedding-1",
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            entity_type=EntityType.CLIP,
            entity_id="clip-1",
            video_id="L21_V001",
            shot_id="shot-1",
            start_ms=0,
            end_ms=1000,
            status=EmbeddingStatus.SUCCESS,
            run_id="run-1",
        )

        self.assertEqual(record.entity_type, EntityType.CLIP)
        self.assertEqual(record.status, EmbeddingStatus.SUCCESS)

    def test_deterministic_id_is_stable(self) -> None:
        first = deterministic_id("dataset", "video", "shot", 0, 1000, "v1")
        second = deterministic_id("dataset", "video", "shot", 0, 1000, "v1")
        changed = deterministic_id("dataset", "video", "shot", 0, 2000, "v1")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
