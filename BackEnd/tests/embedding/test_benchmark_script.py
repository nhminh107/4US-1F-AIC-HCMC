from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import EmbeddingRecord
from BackEnd.app.embedding import CONFIG
from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
from BackEnd.app.embedding.scripts.benchmark_embeddings import summarize_artifact


class BenchmarkScriptTests(unittest.TestCase):
    def test_summarize_artifact_reports_norm_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-1",
                config=ArtifactWriterConfig(output_root=Path(temp_dir)),
            )
            record = EmbeddingRecord(
                embedding_id="embedding-1",
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                entity_type=CONFIG.EntityType.CLIP,
                entity_id="clip-1",
                video_id="L21_V001",
                shot_id="shot-1",
                start_ms=0,
                end_ms=1000,
                dimension=2,
                status=CONFIG.EmbeddingStatus.SUCCESS,
                run_id="run-1",
            )
            manifest = writer.write(np.array([[1, 0]], dtype=np.float32), [record])
            manifest_path = Path(temp_dir) / CLIP_EMBEDDING_SPACE_ID / "run-1" / "manifest.json"

            summary = summarize_artifact(manifest_path)

            self.assertTrue(summary["valid"])
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["mean_norm"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

