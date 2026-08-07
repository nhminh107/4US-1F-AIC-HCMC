from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import EmbeddingRecord
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding import CONFIG
from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig
from BackEnd.app.embedding.artifacts.validator import validate_embedding_artifact
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
from BackEnd.app.embedding.shot.service import SHOT_EMBEDDING_SPACE_ID, ShotEmbeddingService


class ShotEmbeddingServiceTests(unittest.TestCase):
    def test_aggregates_clip_artifact_into_shot_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            clip_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="clip-run",
                config=ArtifactWriterConfig(output_root=output_root),
            )
            clip_records = [
                self._clip_record("embedding-a", "clip-a", 0, 10, 0),
                self._clip_record("embedding-b", "clip-b", 10, 30, 1),
            ]
            clip_manifest = clip_writer.write(
                np.array([[1, 0], [0, 1]], dtype=np.float32),
                clip_records,
            )
            clip_root = output_root / CLIP_EMBEDDING_SPACE_ID / "clip-run"

            shot_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.SHOT,
                embedding_space_id=SHOT_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="shot-run",
                config=ArtifactWriterConfig(output_root=output_root),
            )
            service = ShotEmbeddingService(run_id="shot-run", artifact_writer=shot_writer)

            manifest = service.aggregate_from_clip_artifact(
                shots=[ShotMetadata("shot-1", "L21_V001", 0, 0, 30)],
                clip_manifest=clip_manifest,
                clip_artifact_root=clip_root,
            )

            self.assertEqual(manifest.success_count, 1)
            shot_root = output_root / SHOT_EMBEDDING_SPACE_ID / "shot-run"
            report = validate_embedding_artifact(manifest, shot_root)
            self.assertTrue(report["valid"], report["errors"])
            vector = np.load(shot_root / manifest.vector_shards[0])[0]
            expected = np.array([10, 20], dtype=np.float32)
            expected = expected / np.linalg.norm(expected)
            np.testing.assert_allclose(vector, expected, atol=1e-6)

    def test_uses_missing_clip_embedder_for_zero_clip_shot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            empty_clip_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="clip-run",
                config=ArtifactWriterConfig(output_root=output_root),
            )
            clip_manifest = empty_clip_writer.write(
                np.empty((0, 2), dtype=np.float32),
                [],
            )
            clip_root = output_root / CLIP_EMBEDDING_SPACE_ID / "clip-run"

            generated_record = self._clip_record("embedding-generated", "clip-generated", 0, 10, 0)

            def missing_clip_embedder(_shot):
                return [generated_record], np.array([[1, 0]], dtype=np.float32)

            shot_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.SHOT,
                embedding_space_id=SHOT_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="shot-run",
                config=ArtifactWriterConfig(output_root=output_root),
            )
            service = ShotEmbeddingService(
                run_id="shot-run",
                artifact_writer=shot_writer,
                missing_clip_embedder=missing_clip_embedder,
            )

            manifest = service.aggregate_from_clip_artifact(
                shots=[ShotMetadata("shot-1", "L21_V001", 0, 0, 10)],
                clip_manifest=clip_manifest,
                clip_artifact_root=clip_root,
            )

            self.assertEqual(manifest.success_count, 1)
            self.assertEqual(manifest.failure_count, 0)

    def test_rejects_incompatible_clip_artifact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            clip_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="clip-run",
                config=ArtifactWriterConfig(output_root=output_root),
            )
            bad_record = self._clip_record("embedding-a", "clip-a", 0, 10, 0)
            bad_record = replace(bad_record, sampling_version="other-sampling@1.0.0")
            clip_manifest = clip_writer.write(
                np.array([[1, 0]], dtype=np.float32),
                [bad_record],
            )
            clip_manifest = replace(
                clip_manifest,
                sampling_version=CONFIG.CLIP_SAMPLING_VERSION,
            )
            clip_root = output_root / CLIP_EMBEDDING_SPACE_ID / "clip-run"
            service = ShotEmbeddingService(run_id="shot-run")

            with self.assertRaisesRegex(ValueError, "sampling version"):
                service.aggregate_from_clip_artifact(
                    shots=[ShotMetadata("shot-1", "L21_V001", 0, 0, 10)],
                    clip_manifest=clip_manifest,
                    clip_artifact_root=clip_root,
                )

    def _clip_record(
        self,
        embedding_id: str,
        clip_id: str,
        start_ms: int,
        end_ms: int,
        vector_row: int,
    ) -> EmbeddingRecord:
        return EmbeddingRecord(
            embedding_id=embedding_id,
            embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
            entity_type=CONFIG.EntityType.CLIP,
            entity_id=clip_id,
            video_id="L21_V001",
            shot_id="shot-1",
            start_ms=start_ms,
            end_ms=end_ms,
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            sampling_version=CONFIG.CLIP_SAMPLING_VERSION,
            aggregation_version=CONFIG.CLIP_AGGREGATION_VERSION,
            dimension=2,
            normalized=True,
            vector_norm=1.0,
            vector_row=vector_row,
            status=CONFIG.EmbeddingStatus.SUCCESS,
            run_id="clip-run",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
