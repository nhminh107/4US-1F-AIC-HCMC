from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import (
    ClipRecord,
    DecodedFrameBatch,
    EmbeddingRecord,
    VideoAsset,
)
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding.CONFIG import ArtifactWriterConfig
from BackEnd.app.embedding.artifacts.reader import load_success_records
from BackEnd.app.embedding.artifacts.validator import validate_embedding_artifact
from BackEnd.app.embedding.clip.builder import build_clips
from BackEnd.app.embedding.clip.service import ClipEmbeddingService


class FakeDecoder:
    def __init__(self) -> None:
        self.decoded_timestamps: list[int] = []

    def decode_nearest_frames(self, video_asset, timestamps_ms):
        self.decoded_timestamps.extend(timestamps_ms)
        return DecodedFrameBatch(
            video_id=video_asset.video_id,
            images=tuple(f"image-{timestamp}" for timestamp in timestamps_ms),
            requested_timestamps_ms=tuple(timestamps_ms),
            actual_timestamps_ms=tuple(timestamps_ms),
            decode_statuses=tuple("success" for _ in timestamps_ms),
        )


class FakeAdapter:
    def metadata(self):
        from BackEnd.app.contracts.embedding import ModelMetadata

        return ModelMetadata(
            model_backend="sentence_transformers",
            model_name="clip-ViT-B-32",
            model_id="sentence-transformers/clip-ViT-B-32",
            model_revision=None,
            dimension=2,
        )

    def encode_images(self, images):
        vectors = []
        for image in images:
            timestamp = int(str(image).split("-")[-1])
            vectors.append([float(timestamp + 1), 1.0])
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_clip(clip_id: str, start_ms: int, end_ms: int) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        video_id="L21_V001",
        shot_id="shot-1",
        start_ms=start_ms,
        end_ms=end_ms,
        scale_type="fixed_window",
        target_num_frames=4,
        sampling_strategy="uniform_midpoint",
        sampling_version="clip-sampling@1.0.0",
        clip_builder_version="clip-builder@1.0.0",
    )


class ClipEmbeddingServiceTests(unittest.TestCase):
    def test_builder_contract_runs_directly_in_embedding_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "L21_V001.mp4"
            video_path.write_bytes(b"fake")
            shots = [
                ShotMetadata(
                    shot_id="shot-1",
                    video_id="L21_V001",
                    shot_index=0,
                    start_ms=0,
                    end_ms=25_000,
                )
            ]

            clips = build_clips(shots)
            service = ClipEmbeddingService(
                decoder=FakeDecoder(),
                model_adapter=FakeAdapter(),
                run_id="pipeline-contract-run",
            )
            vectors, records = service.embed_clips_to_matrix(
                clips,
                {"L21_V001": VideoAsset("L21_V001", video_path)},
            )

            self.assertTrue(all(isinstance(clip, ClipRecord) for clip in clips))
            self.assertTrue(
                all(isinstance(record, EmbeddingRecord) for record in records)
            )
            self.assertEqual(vectors.shape, (len(clips), 2))

    def test_service_decodes_unique_frames_and_writes_clip_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            decoder = FakeDecoder()
            service = ClipEmbeddingService(
                decoder=decoder,
                model_adapter=FakeAdapter(),
                run_id="run-1",
            )
            video_path = Path(temp_dir) / "L21_V001.mp4"
            video_path.write_bytes(b"fake")
            clips = [make_clip("a", 0, 1000), make_clip("b", 0, 1000)]

            from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
            from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
            from BackEnd.app.embedding import CONFIG

            service.artifact_writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-1",
                config=ArtifactWriterConfig(output_root=Path(temp_dir) / "artifacts"),
            )

            manifest = service.embed_clips(
                clips,
                {"L21_V001": VideoAsset("L21_V001", video_path)},
            )

            self.assertEqual(decoder.decoded_timestamps, [125, 375, 625, 875])
            self.assertEqual(manifest.success_count, 2)
            artifact_root = Path(temp_dir) / "artifacts" / manifest.embedding_space_id / "run-1"
            report = validate_embedding_artifact(manifest, artifact_root)
            self.assertTrue(report["valid"], report["errors"])

    def test_service_writes_failure_only_artifact_when_video_asset_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
            from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
            from BackEnd.app.embedding import CONFIG

            writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-1",
                config=ArtifactWriterConfig(output_root=Path(temp_dir) / "artifacts"),
            )
            service = ClipEmbeddingService(
                decoder=FakeDecoder(),
                model_adapter=FakeAdapter(),
                run_id="run-1",
                artifact_writer=writer,
            )

            manifest = service.embed_clips([make_clip("a", 0, 1000)], {})

            self.assertEqual(manifest.success_count, 0)
            self.assertEqual(manifest.failure_count, 1)

    def test_service_propagates_adapter_metadata_failure(self) -> None:
        class BrokenMetadataAdapter:
            def metadata(self):
                raise RuntimeError("metadata failed")

            def encode_images(self, images):
                return np.empty((0, 2), dtype=np.float32)

        service = ClipEmbeddingService(
            decoder=FakeDecoder(),
            model_adapter=BrokenMetadataAdapter(),
            run_id="run-1",
        )

        with self.assertRaisesRegex(RuntimeError, "metadata failed"):
            service.embed_clips([], {})

    def test_service_records_actual_timestamps_and_applies_tolerance(self) -> None:
        class OffsetDecoder:
            def decode_nearest_frames(self, video_asset, timestamps_ms):
                return DecodedFrameBatch(
                    video_id=video_asset.video_id,
                    images=tuple(f"image-{timestamp}" for timestamp in timestamps_ms),
                    requested_timestamps_ms=tuple(timestamps_ms),
                    actual_timestamps_ms=tuple(timestamp + 10 for timestamp in timestamps_ms),
                    decode_statuses=tuple("success" for _ in timestamps_ms),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
            from BackEnd.app.embedding.clip.service import CLIP_EMBEDDING_SPACE_ID
            from BackEnd.app.embedding import CONFIG

            video_path = Path(temp_dir) / "L21_V001.mp4"
            video_path.write_bytes(b"fake")
            writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-1",
                config=ArtifactWriterConfig(output_root=Path(temp_dir) / "artifacts"),
            )
            service = ClipEmbeddingService(
                decoder=OffsetDecoder(),
                model_adapter=FakeAdapter(),
                run_id="run-1",
                artifact_writer=writer,
                decode_tolerance_ms=20,
            )

            manifest = service.embed_clips(
                [make_clip("a", 0, 1000)],
                {"L21_V001": VideoAsset("L21_V001", video_path)},
            )

            artifact_root = Path(temp_dir) / "artifacts" / manifest.embedding_space_id / "run-1"
            records = load_success_records(artifact_root, manifest)
            self.assertEqual(manifest.success_count, 1)
            self.assertEqual(records[0].actual_timestamps_ms, tuple(timestamp + 10 for timestamp in records[0].sampled_timestamps_ms))

            writer = EmbeddingArtifactWriter(
                entity_type=CONFIG.EntityType.CLIP,
                embedding_space_id=CLIP_EMBEDDING_SPACE_ID,
                model_backend=CONFIG.CLIP_BACKEND,
                model_name=CONFIG.CLIP_MODEL,
                dimension=2,
                run_id="run-2",
                config=ArtifactWriterConfig(output_root=Path(temp_dir) / "artifacts"),
            )
            strict_service = ClipEmbeddingService(
                decoder=OffsetDecoder(),
                model_adapter=FakeAdapter(),
                run_id="run-2",
                artifact_writer=writer,
                decode_tolerance_ms=1,
            )

            strict_manifest = strict_service.embed_clips(
                [make_clip("b", 0, 1000)],
                {"L21_V001": VideoAsset("L21_V001", video_path)},
            )

            self.assertEqual(strict_manifest.success_count, 0)
            self.assertEqual(strict_manifest.failure_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
