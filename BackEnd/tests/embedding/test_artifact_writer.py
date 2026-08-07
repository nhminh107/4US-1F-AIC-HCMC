from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import EmbeddingRecord, EmbeddingStatus, EntityType
from BackEnd.app.embedding.artifacts.validator import validate_embedding_artifact
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig


class ArtifactWriterTests(unittest.TestCase):
    def test_writes_vector_metadata_failure_and_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ArtifactWriterConfig(output_root=Path(temp_dir))
            success = EmbeddingRecord(
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
            failure = EmbeddingRecord(
                embedding_id="embedding-2",
                embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                entity_type=EntityType.CLIP,
                entity_id="clip-2",
                video_id="L21_V001",
                status=EmbeddingStatus.DECODE_FAILED,
                run_id="run-1",
                error_message="decode failed",
            )
            writer = EmbeddingArtifactWriter(
                entity_type=EntityType.CLIP,
                embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                model_backend="sentence_transformers",
                model_name="clip-ViT-B-32",
                dimension=2,
                run_id="run-1",
                config=config,
            )

            manifest = writer.write(
                np.array([[1, 0]], dtype=np.float32),
                [success, failure],
            )

            artifact_root = Path(temp_dir) / manifest.embedding_space_id / manifest.run_id
            self.assertTrue((artifact_root / "manifest.json").is_file())
            report = validate_embedding_artifact(manifest, artifact_root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(manifest.success_count, 1)
            self.assertEqual(manifest.failure_count, 1)

    def test_writes_failure_only_artifact_with_empty_vector_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ArtifactWriterConfig(output_root=Path(temp_dir))
            failure = EmbeddingRecord(
                embedding_id="embedding-2",
                embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                entity_type=EntityType.CLIP,
                entity_id="clip-2",
                video_id="L21_V001",
                status=EmbeddingStatus.MEDIA_NOT_FOUND,
                run_id="run-1",
                error_message="missing video",
            )
            writer = EmbeddingArtifactWriter(
                entity_type=EntityType.CLIP,
                embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                model_backend="sentence_transformers",
                model_name="clip-ViT-B-32",
                dimension=2,
                run_id="run-1",
                config=config,
            )

            manifest = writer.write(np.empty((0, 2), dtype=np.float32), [failure])

            artifact_root = Path(temp_dir) / manifest.embedding_space_id / manifest.run_id
            self.assertEqual(manifest.success_count, 0)
            self.assertEqual(manifest.failure_count, 1)
            self.assertEqual(np.load(artifact_root / manifest.vector_shards[0]).shape, (0, 2))
            report = validate_embedding_artifact(manifest, artifact_root)
            self.assertTrue(report["valid"], report["errors"])

    def test_shards_vectors_by_rows_per_shard_and_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ArtifactWriterConfig(output_root=Path(temp_dir), rows_per_shard=2)
            records = [
                EmbeddingRecord(
                    embedding_id=f"embedding-{index}",
                    embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                    entity_type=EntityType.CLIP,
                    entity_id=f"clip-{index}",
                    video_id="L21_V001",
                    status=EmbeddingStatus.SUCCESS,
                    run_id="run-1",
                )
                for index in range(3)
            ]
            writer = EmbeddingArtifactWriter(
                entity_type=EntityType.CLIP,
                embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
                model_backend="sentence_transformers",
                model_name="clip-ViT-B-32",
                dimension=2,
                run_id="run-1",
                config=config,
            )

            manifest = writer.write(np.eye(3, 2, dtype=np.float32), records)

            self.assertEqual(manifest.shard_count, 2)
            self.assertEqual(len(manifest.vector_shards), 2)
            with self.assertRaises(FileExistsError):
                writer.write(np.eye(3, 2, dtype=np.float32), records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
