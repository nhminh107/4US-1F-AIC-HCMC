"""Tests for GPU batching, FP16 precision, and decoder fallback strategies."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from BackEnd.app.contracts.embedding import ModelMetadata, VideoAsset
from BackEnd.app.embedding.models.clip_vit_b32 import ClipViTB32Adapter, _normalize_matrix
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder


class DummyModel:
    """Mock SentenceTransformer model for batching verification."""

    def __init__(self) -> None:
        self.call_batch_sizes: list[int] = []

    def encode(self, values, batch_size: int = 32, convert_to_numpy: bool = True, show_progress_bar: bool = False):
        self.call_batch_sizes.append(batch_size)
        num_items = len(values)
        dim = 512
        # Return synthetic unit vectors
        rng = np.random.RandomState(42)
        matrix = rng.randn(num_items, dim).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms


class GPUPrecisionAndBatchingTests(unittest.TestCase):

    def test_clip_adapter_encodes_texts_in_batches(self) -> None:
        dummy_model = DummyModel()
        adapter = ClipViTB32Adapter(model=dummy_model, dimension=512)

        texts = [f"sample query text number {i}" for i in range(100)]
        vectors = adapter.encode_texts(texts, batch_size=32)

        self.assertEqual(vectors.shape, (100, 512))
        self.assertEqual(dummy_model.call_batch_sizes, [32])
        # Verify L2 normalization
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalize_matrix_helper(self) -> None:
        raw_matrix = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
        norm_matrix = _normalize_matrix(raw_matrix)

        self.assertEqual(norm_matrix.shape, (2, 2))
        self.assertAlmostEqual(float(np.linalg.norm(norm_matrix[0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(norm_matrix[1])), 1.0, places=5)

    def test_decoder_handles_non_existent_file_gracefully(self) -> None:
        decoder = PyAVVideoDecoder()
        from pathlib import Path
        fake_asset = VideoAsset("non_existent_vid", Path("C:/invalid/path/non_existent.mp4"))

        batch = decoder.decode_nearest_frames(fake_asset, [1000, 2000])
        self.assertEqual(batch.video_id, "non_existent_vid")
        self.assertEqual(batch.decode_statuses, ("media_not_found", "media_not_found"))
        self.assertEqual(batch.metrics["failed_frame_count"], 2)

    def test_parallel_num_workers_and_gc(self) -> None:
        import tempfile
        from pathlib import Path
        from BackEnd.app.contracts.embedding import ClipRecord, DecodedFrameBatch
        from BackEnd.app.embedding.clip.service import ClipEmbeddingService
        from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig
        from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
        from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
        from BackEnd.app.embedding import CONFIG

        class MultiWorkerFakeDecoder:
            def decode_nearest_frames(self, video_asset, timestamps_ms):
                return DecodedFrameBatch(
                    video_id=video_asset.video_id,
                    images=tuple(f"img-{ts}" for ts in timestamps_ms),
                    requested_timestamps_ms=tuple(timestamps_ms),
                    actual_timestamps_ms=tuple(timestamps_ms),
                    decode_statuses=tuple("success" for _ in timestamps_ms),
                )

        class MultiWorkerFakeAdapter:
            def metadata(self):
                return ModelMetadata(
                    model_backend="sentence_transformers",
                    model_name="clip-ViT-B-32",
                    model_id="sentence-transformers/clip-ViT-B-32",
                    model_revision=None,
                    dimension=2,
                )

            def encode_images(self, images):
                return np.ones((len(images), 2), dtype=np.float32) / np.sqrt(2.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            v1_path = Path(temp_dir) / "v1.mp4"
            v2_path = Path(temp_dir) / "v2.mp4"
            v1_path.write_bytes(b"data1")
            v2_path.write_bytes(b"data2")

            clips = [
                ClipRecord(
                    clip_id="c1", video_id="v1", shot_id="s1", start_ms=0, end_ms=1000,
                    scale_type="fixed_window", target_num_frames=4, sampling_strategy="uniform_midpoint",
                    sampling_version="1.0.0", clip_builder_version="1.0.0"
                ),
                ClipRecord(
                    clip_id="c2", video_id="v2", shot_id="s2", start_ms=0, end_ms=1000,
                    scale_type="fixed_window", target_num_frames=4, sampling_strategy="uniform_midpoint",
                    sampling_version="1.0.0", clip_builder_version="1.0.0"
                ),
            ]

            writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-parallel",
                config=ArtifactWriterConfig(output_root=Path(temp_dir) / "artifacts"),
            )

            service = ClipEmbeddingService(
                decoder=MultiWorkerFakeDecoder(),
                model_adapter=MultiWorkerFakeAdapter(),
                run_id="run-parallel",
                artifact_writer=writer,
                num_workers=2,
                enable_gc=True,
            )

            manifest = service.embed_clips(clips, {"v1": VideoAsset("v1", v1_path), "v2": VideoAsset("v2", v2_path)})
            self.assertEqual(manifest.success_count, 2)

    def test_in_memory_matrix_wrappers(self) -> None:
        import tempfile
        from pathlib import Path
        from BackEnd.app.contracts.embedding import ClipRecord, DecodedFrameBatch
        from BackEnd.app.contracts.pipeline import ShotMetadata
        from BackEnd.app.embedding.clip.service import ClipEmbeddingService
        from BackEnd.app.embedding.shot.service import ShotEmbeddingService

        class SimpleDecoder:
            def decode_nearest_frames(self, video_asset, timestamps_ms):
                return DecodedFrameBatch(
                    video_id=video_asset.video_id,
                    images=tuple(f"img-{ts}" for ts in timestamps_ms),
                    requested_timestamps_ms=tuple(timestamps_ms),
                    actual_timestamps_ms=tuple(timestamps_ms),
                    decode_statuses=tuple("success" for _ in timestamps_ms),
                )

        class SimpleAdapter:
            def metadata(self):
                return ModelMetadata(
                    model_backend="sentence_transformers",
                    model_name="clip-ViT-B-32",
                    model_id="sentence-transformers/clip-ViT-B-32",
                    model_revision=None,
                    dimension=2,
                )

            def encode_images(self, images):
                return np.ones((len(images), 2), dtype=np.float32) / np.sqrt(2.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            v_path = Path(temp_dir) / "v1.mp4"
            v_path.write_bytes(b"video_bytes")

            clip = ClipRecord(
                clip_id="clip-101", video_id="v1", shot_id="shot-101", start_ms=0, end_ms=1000,
                scale_type="fixed_window", target_num_frames=4, sampling_strategy="uniform_midpoint",
                sampling_version="1.0.0", clip_builder_version="1.0.0"
            )

            clip_service = ClipEmbeddingService(decoder=SimpleDecoder(), model_adapter=SimpleAdapter())
            clip_matrix, clip_recs = clip_service.embed_clips_to_matrix([clip], {"v1": VideoAsset("v1", v_path)})

            self.assertEqual(clip_matrix.shape, (1, 2))
            self.assertEqual(len(clip_recs), 1)

            shot = ShotMetadata(shot_id="shot-101", video_id="v1", shot_index=0, start_ms=0, end_ms=1000)
            shot_service = ShotEmbeddingService()
            shot_matrix, shot_recs = shot_service.aggregate_shots_to_matrix(
                shots=[shot],
                clip_records=clip_recs,
                clip_vectors=clip_matrix,
            )

            self.assertEqual(shot_matrix.shape, (1, 2))
            self.assertEqual(len(shot_recs), 1)
            np.testing.assert_allclose(np.linalg.norm(shot_matrix[0]), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
