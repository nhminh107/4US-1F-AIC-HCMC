"""Objective end-to-end integration test suite using real competition dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from BackEnd.app.contracts.embedding import (
    ClipRecord,
    EmbeddingArtifactManifest,
    EmbeddingStatus,
    EntityType,
    VideoAsset,
)
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding import CONFIG
from BackEnd.app.embedding.artifacts.writer import EmbeddingArtifactWriter
from BackEnd.app.embedding.artifacts.reader import load_vectors
from BackEnd.app.embedding.artifacts.validator import load_manifest, validate_embedding_artifact
from BackEnd.app.embedding.clip.builder import build_clips
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder
from BackEnd.app.embedding.clip.service import ClipEmbeddingService
from BackEnd.app.embedding.clip.video_repository import VideoRepository
from BackEnd.app.embedding.evaluation.exact_retrieval import exact_top_k
from BackEnd.app.embedding.evaluation.text_encoder import encode_clip_queries
from BackEnd.app.embedding.models.clip_vit_b32 import ClipViTB32Adapter
from BackEnd.app.embedding.shot.service import ShotEmbeddingService

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
REAL_VIDEO_DIR = DATA_DIR / "Videos_L21_a" / "video"
REAL_VIDEO_PATH = REAL_VIDEO_DIR / "L21_V001.mp4"
REAL_MAP_KEYFRAME_PATH = DATA_DIR / "map-keyframes-aic25-b1" / "map-keyframes" / "L21_V001.csv"
REAL_MEDIA_INFO_PATH = DATA_DIR / "media-info-aic25-b1" / "media-info" / "L21_V001.json"


class RealDataIntegrationTests(unittest.TestCase):
    """Objective tests against real video files and metadata without mocks."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Group 1: VideoRepository on Real Data
    # -------------------------------------------------------------------------
    def test_group1_default_video_repository_on_real_data(self) -> None:
        """Test if default VideoRepository resolves real L21_V001.mp4."""
        repo = VideoRepository()  # Uses DEFAULT_VIDEO_DIR = data/video
        asset = repo.resolve_video("L21_V001")
        self.assertTrue(asset.video_uri.is_file())

    def test_group1_configured_video_repository_on_real_data(self) -> None:
        """Test VideoRepository with explicit real video directory."""
        if not REAL_VIDEO_DIR.is_dir():
            self.skipTest(f"Real video directory not found: {REAL_VIDEO_DIR}")
        repo = VideoRepository(video_dir=REAL_VIDEO_DIR)
        asset = repo.resolve_video("L21_V001")
        self.assertEqual(asset.video_id, "L21_V001")
        self.assertTrue(asset.video_uri.is_file())

    def test_group1_video_repository_non_existent_video(self) -> None:
        """Test resolving a non-existent video ID raises MediaNotFoundError."""
        repo = VideoRepository(video_dir=REAL_VIDEO_DIR if REAL_VIDEO_DIR.is_dir() else DATA_DIR)
        with self.assertRaises(Exception) as ctx:
            repo.resolve_video("NON_EXISTENT_VIDEO_99999")
        self.assertIn("MediaNotFoundError", type(ctx.exception).__name__)

    # -------------------------------------------------------------------------
    # Group 2: PyAV Video Decoder on Real Video
    # -------------------------------------------------------------------------
    def test_group2_pyav_decode_real_video_valid_timestamps(self) -> None:
        """Test decoding valid timestamps on real L21_V001.mp4."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        decoder = PyAVVideoDecoder()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)
        requested_ms = [0, 1000, 5000, 10000, 20000]

        batch = decoder.decode_nearest_frames(video_asset, requested_ms)

        self.assertEqual(batch.video_id, "L21_V001")
        self.assertEqual(len(batch.images), len(requested_ms))
        self.assertTrue(all(status == "success" for status in batch.decode_statuses))
        self.assertTrue(all(img is not None for img in batch.images))
        self.assertTrue(all(isinstance(actual, int) for actual in batch.actual_timestamps_ms))

    def test_group2_pyav_decode_real_video_extreme_out_of_bounds_timestamp(self) -> None:
        """Test decoding timestamp beyond video length (e.g. 99,999,999 ms)."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        decoder = PyAVVideoDecoder()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)
        requested_ms = [99_999_999]

        batch = decoder.decode_nearest_frames(video_asset, requested_ms)

        self.assertEqual(len(batch.images), 1)
        # Should gracefully handle out of bounds (decode_failed or nearest end frame)
        self.assertIn(batch.decode_statuses[0], ("success", "decode_failed"))

    # -------------------------------------------------------------------------
    # Group 3: Integration with Real Competition Keyframe Mapping CSV
    # -------------------------------------------------------------------------
    def test_group3_keyframe_csv_mapping_and_decoding(self) -> None:
        """Parse real keyframe CSV L21_V001.csv and decode frames at keyframe pts_time."""
        if not REAL_MAP_KEYFRAME_PATH.is_file() or not REAL_VIDEO_PATH.is_file():
            self.skipTest("Real keyframe CSV or video file not found.")

        keyframes = []
        with REAL_MAP_KEYFRAME_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pts_time_sec = float(row["pts_time"])
                timestamp_ms = int(round(pts_time_sec * 1000))
                keyframes.append((int(row["n"]), timestamp_ms))

        self.assertGreater(len(keyframes), 0)

        # Sample first 5 keyframe timestamps
        sample_timestamps = [ms for _, ms in keyframes[:5]]
        decoder = PyAVVideoDecoder()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)
        batch = decoder.decode_nearest_frames(video_asset, sample_timestamps)

        self.assertEqual(sum(1 for s in batch.decode_statuses if s == "success"), len(sample_timestamps))

    # -------------------------------------------------------------------------
    # Group 4: Real Shot & Clip Building
    # -------------------------------------------------------------------------
    def test_group4_clip_builder_with_real_video_duration(self) -> None:
        """Build clip records for shots derived from real video metadata."""
        if not REAL_MEDIA_INFO_PATH.is_file():
            self.skipTest(f"Real media info JSON not found: {REAL_MEDIA_INFO_PATH}")

        with REAL_MEDIA_INFO_PATH.open("r", encoding="utf-8") as f:
            info = json.load(f)
        video_length_sec = float(info.get("length", 100))
        video_duration_ms = int(video_length_sec * 1000)

        # Create shots: 1 short (5s), 1 standard (15s), 1 long (60s)
        shots = [
            ShotMetadata(video_id="L21_V001", shot_id="shot-1", shot_index=0, start_ms=0, end_ms=5000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-2", shot_index=1, start_ms=5000, end_ms=20000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-3", shot_index=2, start_ms=20000, end_ms=80000),
        ]

        config = CONFIG.ClipBuilderConfig(dataset_id="aic_hcm2026")
        clips = build_clips(shots, config)

        self.assertGreater(len(clips), 0)
        # Shot 1 <= 10s => 1 clip (full_shot)
        shot1_clips = [c for c in clips if c.shot_id == "shot-1"]
        self.assertEqual(len(shot1_clips), 1)
        self.assertEqual(shot1_clips[0].scale_type, "full_shot")

        # Shot 3 (60s) => fixed_window clips
        shot3_clips = [c for c in clips if c.shot_id == "shot-3"]
        self.assertGreater(len(shot3_clips), 1)

    def test_group4_shot_extending_past_real_video_length(self) -> None:
        """Test shot definition extending past real video length."""
        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-out-bounds", shot_index=0, start_ms=1000, end_ms=999_999)
        clips = build_clips([shot])
        self.assertGreater(len(clips), 0)
        # Builder creates logical clips based on shot bounds without failing
        self.assertTrue(all(c.start_ms >= 1000 for c in clips))

    # -------------------------------------------------------------------------
    # Group 5: Real End-to-End Clip Embedding Service
    # -------------------------------------------------------------------------
    def test_group5_clip_embedding_service_on_real_video(self) -> None:
        """Run ClipEmbeddingService end-to-end on real L21_V001.mp4 video."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        # Real dependencies
        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        # Create clips for first 30 seconds
        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-real-1", shot_index=0, start_ms=0, end_ms=30000)
        clips = build_clips([shot])

        custom_writer_config = CONFIG.ArtifactWriterConfig(output_root=self.temp_dir)
        writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="real-data-run-1",
            config=custom_writer_config,
        )

        service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="real-data-run-1",
            artifact_writer=writer,
        )

        manifest = service.embed_clips(clips, {"L21_V001": video_asset})

        # Verify output manifest
        self.assertEqual(manifest.entity_type, EntityType.CLIP)
        self.assertEqual(manifest.record_count, len(clips))
        self.assertGreater(manifest.success_count, 0)
        self.assertEqual(manifest.dimension, 512)

        # Load vectors and verify norm
        artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "real-data-run-1"
        validation = validate_embedding_artifact(manifest, artifact_root)
        self.assertTrue(validation["valid"])

        vectors = load_vectors(artifact_root, manifest)
        self.assertEqual(vectors.shape, (manifest.success_count, 512))
        norms = np.linalg.norm(vectors, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    # -------------------------------------------------------------------------
    # Group 6: Real End-to-End Shot Embedding Service
    # -------------------------------------------------------------------------
    def test_group6_shot_embedding_service_on_real_clip_artifact(self) -> None:
        """Aggregate real clip artifact into a shot embedding artifact."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        # Step 1: Create real clip artifact
        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        shots = [
            ShotMetadata(video_id="L21_V001", shot_id="shot-s1", shot_index=0, start_ms=0, end_ms=15000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-s2", shot_index=1, start_ms=15000, end_ms=30000),
        ]
        clips = build_clips(shots)

        clip_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="real-clip-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        clip_service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="real-clip-run",
            artifact_writer=clip_writer,
        )
        clip_manifest = clip_service.embed_clips(clips, {"L21_V001": video_asset})

        # Step 2: Aggregate shots using ShotEmbeddingService
        clip_artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "real-clip-run"
        shot_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.SHOT,
            embedding_space_id="shot.clip_vit_b32.coverage_pool_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="real-shot-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        shot_service = ShotEmbeddingService(
            run_id="real-shot-run",
            artifact_writer=shot_writer,
        )

        shot_manifest = shot_service.aggregate_from_clip_artifact(
            shots=shots,
            clip_manifest=clip_manifest,
            clip_artifact_root=clip_artifact_root,
        )

        self.assertEqual(shot_manifest.entity_type, EntityType.SHOT)
        self.assertEqual(shot_manifest.success_count, len(shots))

        shot_artifact_root = self.temp_dir / "shot.clip_vit_b32.coverage_pool_v1" / "real-shot-run"
        shot_vectors = load_vectors(shot_artifact_root, shot_manifest)
        self.assertEqual(shot_vectors.shape, (len(shots), 512))
        shot_norms = np.linalg.norm(shot_vectors, axis=1)
        self.assertTrue(np.allclose(shot_norms, 1.0, atol=1e-3))

    # -------------------------------------------------------------------------
    # Group 7: Real Exact Retrieval Query
    # -------------------------------------------------------------------------
    def test_group7_exact_retrieval_query_on_real_video_vectors(self) -> None:
        """Encode queries and retrieve Top-K from real video visual vectors."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        # Generate real visual vectors for L21_V001
        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-q", shot_index=0, start_ms=0, end_ms=20000)
        clips = build_clips([shot])

        writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="retrieval-test-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="retrieval-test-run",
            artifact_writer=writer,
        )
        manifest = service.embed_clips(clips, {"L21_V001": video_asset})

        artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "retrieval-test-run"
        visual_vectors = load_vectors(artifact_root, manifest)

        # Encode queries (English and Vietnamese)
        queries = [
            "television news presenter speaking in studio",
            "bản tin thời sự 60 giây",
        ]
        query_vectors = encode_clip_queries(queries, adapter=adapter)

        self.assertEqual(query_vectors.shape, (2, 512))

        # Perform exact matrix top-k
        scores, top_indexes = exact_top_k(query_vectors, visual_vectors, top_k=2)

        self.assertEqual(scores.shape, (2, min(2, len(clips))))
        self.assertEqual(top_indexes.shape, (2, min(2, len(clips))))
        self.assertTrue(np.all(scores <= 1.0 + 1e-5))

    def test_group4_clip_builder_overlapping_and_adjacent_shots(self) -> None:
        """Test building clips for adjacent & overlapping shots on real video."""
        shots = [
            ShotMetadata(video_id="L21_V001", shot_id="shot-0", shot_index=0, start_ms=0, end_ms=10000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-1", shot_index=1, start_ms=8000, end_ms=18000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-2", shot_index=2, start_ms=18000, end_ms=35000),
        ]
        clips = build_clips(shots)

        self.assertGreater(len(clips), 3)
        # Check determinism: re-building produces identical clip IDs
        clips_rebuilt = build_clips(shots)
        self.assertEqual([c.clip_id for c in clips], [c.clip_id for c in clips_rebuilt])

    def test_group5_clip_embedding_chunking_max_clips_per_unit(self) -> None:
        """Test work planner chunking work units with max_clips_per_unit."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        from BackEnd.app.embedding.clip.planner import plan_video_work

        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-long", shot_index=0, start_ms=0, end_ms=45000)
        clips = build_clips([shot])

        # Verify planner chunks work into units of max 2 clips each
        work_units = plan_video_work(clips, {"L21_V001": video_asset}, max_clips_per_unit=2)
        self.assertGreater(len(work_units), 1)
        self.assertTrue(all(len(wu.sorted_clip_records) <= 2 for wu in work_units))

        writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="chunking-test-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="chunking-test-run",
            artifact_writer=writer,
        )

        manifest = service.embed_clips(clips, {"L21_V001": video_asset})
        self.assertEqual(manifest.record_count, len(clips))
        self.assertEqual(manifest.success_count, len(clips))

    def test_group6_shot_aggregation_zero_clip_fallback_on_real_data(self) -> None:
        """Test ShotEmbeddingService zero-clip fallback behavior on real data."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        # Shot 1 has clips, Shot 2 has NO clips in manifest (zero-clip fallback)
        shots = [
            ShotMetadata(video_id="L21_V001", shot_id="shot-valid", shot_index=0, start_ms=0, end_ms=10000),
            ShotMetadata(video_id="L21_V001", shot_id="shot-missing-clips", shot_index=1, start_ms=90000, end_ms=100000),
        ]
        valid_clips = build_clips([shots[0]])  # Only build clips for shot 0

        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        clip_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="zeroclip-clip-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        clip_service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="zeroclip-clip-run",
            artifact_writer=clip_writer,
        )
        clip_manifest = clip_service.embed_clips(valid_clips, {"L21_V001": video_asset})

        shot_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.SHOT,
            embedding_space_id="shot.clip_vit_b32.coverage_pool_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="zeroclip-shot-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        shot_service = ShotEmbeddingService(
            run_id="zeroclip-shot-run",
            artifact_writer=shot_writer,
        )

        clip_artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "zeroclip-clip-run"
        shot_manifest = shot_service.aggregate_from_clip_artifact(
            shots=shots,
            clip_manifest=clip_manifest,
            clip_artifact_root=clip_artifact_root,
        )

        self.assertEqual(shot_manifest.record_count, 2)
        self.assertEqual(shot_manifest.success_count, 1)
        self.assertEqual(shot_manifest.failure_count, 1)

    def test_group7_exact_retrieval_batch_queries_and_ranking(self) -> None:
        """Batch retrieval test with 5 queries including negative/unrelated query."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-batch", shot_index=0, start_ms=0, end_ms=30000)
        clips = build_clips([shot])

        writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="batch-retrieval-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="batch-retrieval-run",
            artifact_writer=writer,
        )
        manifest = service.embed_clips(clips, {"L21_V001": video_asset})

        artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "batch-retrieval-run"
        visual_vectors = load_vectors(artifact_root, manifest)

        queries = [
            "news presenter in studio",
            "television broadcast studio host",
            "bản tin thời sự phát thanh",
            "người dẫn chương trình truyền hình",
            "underwater deep sea submarine fish coral reef",  # Unrelated query
        ]
        query_vectors = encode_clip_queries(queries, adapter=adapter)

        self.assertEqual(query_vectors.shape, (5, 512))

        scores, top_indexes = exact_top_k(query_vectors, visual_vectors, top_k=3)

        self.assertEqual(scores.shape, (5, 3))
        # Unrelated query score should be lower than related studio news queries
        related_max_score = np.max(scores[0])
        unrelated_max_score = np.max(scores[4])
        self.assertGreater(related_max_score, unrelated_max_score)

    def test_group7_shot_level_retrieval_vs_clip_level_retrieval(self) -> None:
        """Compare retrieval scores between Clip-level vectors and Shot-level vectors."""
        if not REAL_VIDEO_PATH.is_file():
            self.skipTest(f"Real video file not found: {REAL_VIDEO_PATH}")

        decoder = PyAVVideoDecoder()
        adapter = ClipViTB32Adapter()
        video_asset = VideoAsset(video_id="L21_V001", video_uri=REAL_VIDEO_PATH)

        shot = ShotMetadata(video_id="L21_V001", shot_id="shot-cmp", shot_index=0, start_ms=0, end_ms=20000)
        clips = build_clips([shot])

        # Clip Artifact
        clip_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.CLIP,
            embedding_space_id="clip.clip_vit_b32.masked_mean16_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="cmp-clip-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        clip_service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=adapter,
            run_id="cmp-clip-run",
            artifact_writer=clip_writer,
        )
        clip_manifest = clip_service.embed_clips(clips, {"L21_V001": video_asset})

        # Shot Artifact
        shot_writer = EmbeddingArtifactWriter(
            entity_type=EntityType.SHOT,
            embedding_space_id="shot.clip_vit_b32.coverage_pool_v1",
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            dimension=512,
            run_id="cmp-shot-run",
            config=CONFIG.ArtifactWriterConfig(output_root=self.temp_dir),
        )
        shot_service = ShotEmbeddingService(
            run_id="cmp-shot-run",
            artifact_writer=shot_writer,
        )
        clip_artifact_root = self.temp_dir / "clip.clip_vit_b32.masked_mean16_v1" / "cmp-clip-run"
        shot_manifest = shot_service.aggregate_from_clip_artifact(
            shots=[shot],
            clip_manifest=clip_manifest,
            clip_artifact_root=clip_artifact_root,
        )

        clip_vectors = load_vectors(clip_artifact_root, clip_manifest)
        shot_artifact_root = self.temp_dir / "shot.clip_vit_b32.coverage_pool_v1" / "cmp-shot-run"
        shot_vectors = load_vectors(shot_artifact_root, shot_manifest)

        query_vec = encode_clip_queries(["news studio broadcast"], adapter=adapter)

        clip_scores, _ = exact_top_k(query_vec, clip_vectors, top_k=1)
        shot_scores, _ = exact_top_k(query_vec, shot_vectors, top_k=1)

        # Both scores should be valid and positive
        self.assertGreater(clip_scores[0, 0], 0.1)
        self.assertGreater(shot_scores[0, 0], 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

