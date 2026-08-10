"""Real-data integration tests for image, clip, and shot embedding stages."""

from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Keep the integration test deterministic and prevent implicit model downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from BackEnd.app.contracts.embedding import (  # noqa: E402
    ClipRecord,
    DecodedFrameBatch,
    EmbeddingStatus,
    ModelMetadata,
    VideoAsset,
)
from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata  # noqa: E402
from BackEnd.app.embedding import CONFIG  # noqa: E402
from BackEnd.app.embedding.ImageEmbedding import ImageEmbedder  # noqa: E402
from BackEnd.app.embedding.clip.service import ClipEmbeddingService  # noqa: E402
from BackEnd.app.embedding.shot.service import ShotEmbeddingService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
VIDEO_ID = "L21_V001"
VIDEO_PATH = DATA_ROOT / "video" / f"{VIDEO_ID}.mp4"
KEYFRAME_DIR = DATA_ROOT / "keyframes" / VIDEO_ID
KEYFRAME_MAP_PATH = DATA_ROOT / "map-keyframes" / f"{VIDEO_ID}.csv"
SHOT_ID = "real-shot-embedding-test"
EMBEDDING_DIMENSION = 512


class RealImageEmbeddingAdapter:
    """Expose the production ImageEmbedder model through the clip protocol."""

    embedding_space_id = "clip.clip_vit_b32.real_data_test"

    def __init__(self, embedder: ImageEmbedder) -> None:
        self.embedder = embedder

    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        vectors = self.embedder.model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.embedder.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def get_dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_backend=CONFIG.CLIP_BACKEND,
            model_name=CONFIG.CLIP_MODEL,
            model_id=CONFIG.CLIP_MODEL_ID,
            model_revision=CONFIG.CLIP_MODEL_REVISION,
            dimension=EMBEDDING_DIMENSION,
            normalized=True,
        )


class RealKeyframeDecoder:
    """Resolve requested timestamps to real keyframe images from the dataset."""

    def __init__(self, map_path: Path, keyframe_dir: Path) -> None:
        self.keyframes: list[tuple[int, Path]] = []
        with map_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                timestamp_ms = int(round(float(row["pts_time"]) * 1_000))
                image_path = keyframe_dir / f"{int(row['n']):03d}.jpg"
                if image_path.is_file():
                    self.keyframes.append((timestamp_ms, image_path))

        if not self.keyframes:
            raise RuntimeError(f"No real keyframes found in {keyframe_dir}")

    def decode_nearest_frames(
        self,
        video_asset: VideoAsset,
        timestamps_ms: list[int] | tuple[int, ...],
    ) -> DecodedFrameBatch:
        images: list[Image.Image] = []
        actual_timestamps: list[int] = []
        for requested_timestamp in timestamps_ms:
            actual_timestamp, image_path = min(
                self.keyframes,
                key=lambda item: abs(item[0] - requested_timestamp),
            )
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
            actual_timestamps.append(actual_timestamp)

        return DecodedFrameBatch(
            video_id=video_asset.video_id,
            images=tuple(images),
            requested_timestamps_ms=tuple(timestamps_ms),
            actual_timestamps_ms=tuple(actual_timestamps),
            decode_statuses=tuple("success" for _ in timestamps_ms),
            metrics={"requested_frame_count": len(timestamps_ms)},
        )


class RealEmbeddingStagesTests(unittest.TestCase):
    """Run the three embedding stages with the real CLIP model and data."""

    embedder: ImageEmbedder
    clip_service: ClipEmbeddingService
    clips: list[ClipRecord]
    shot: ShotMetadata
    video_asset: VideoAsset
    clip_result: tuple[np.ndarray, list] | None = None

    @classmethod
    def setUpClass(cls) -> None:
        required_paths = [VIDEO_PATH, KEYFRAME_MAP_PATH, KEYFRAME_DIR]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(f"Required real embedding data is missing: {missing}")

        cls.embedder = ImageEmbedder()
        decoder = RealKeyframeDecoder(KEYFRAME_MAP_PATH, KEYFRAME_DIR)
        cls.clip_service = ClipEmbeddingService(
            decoder=decoder,
            model_adapter=RealImageEmbeddingAdapter(cls.embedder),
            run_id="real-embedding-stage-test",
            enable_gc=False,
        )
        cls.video_asset = VideoAsset(video_id=VIDEO_ID, video_uri=VIDEO_PATH)

        # These windows have midpoints at real keyframe timestamps 3.0s and 8.7s.
        cls.clips = [
            cls._make_clip("real-clip-1", 2_900, 3_100),
            cls._make_clip("real-clip-2", 8_600, 8_800),
        ]
        cls.shot = ShotMetadata(
            shot_id=SHOT_ID,
            video_id=VIDEO_ID,
            shot_index=0,
            start_ms=2_900,
            end_ms=8_800,
        )

    @staticmethod
    def _make_clip(clip_id: str, start_ms: int, end_ms: int) -> ClipRecord:
        return ClipRecord(
            clip_id=clip_id,
            video_id=VIDEO_ID,
            shot_id=SHOT_ID,
            start_ms=start_ms,
            end_ms=end_ms,
            scale_type="fixed_window",
            target_num_frames=1,
            sampling_strategy=CONFIG.CLIP_SAMPLING_STRATEGY,
            sampling_version=CONFIG.CLIP_SAMPLING_VERSION,
            clip_builder_version=CONFIG.CLIP_BUILDER_VERSION,
        )

    @classmethod
    def _get_clip_result(cls) -> tuple[np.ndarray, list]:
        if cls.clip_result is None:
            cls.clip_result = cls.clip_service.embed_clips_to_matrix(
                cls.clips,
                {VIDEO_ID: cls.video_asset},
            )
        return cls.clip_result

    def test_image_embedding_with_real_keyframe(self) -> None:
        frame = FrameMetadata(
            frame_id=f"{VIDEO_ID}-001",
            video_id=VIDEO_ID,
            shot_id=SHOT_ID,
            timestamp_ms=0,
            fps=30.0,
            frame_idx=0,
            source="official",
            n=1,
            frame_path=KEYFRAME_DIR / "001.jpg",
        )

        vector = self.embedder.embed(frame)

        self.assertEqual(vector.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_clip_embedding_with_real_keyframes(self) -> None:
        vectors, records = self._get_clip_result()

        self.assertEqual(vectors.shape, (2, EMBEDDING_DIMENSION))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1),
            np.ones(2, dtype=np.float32),
            atol=1e-5,
        )
        self.assertEqual(
            [record.status for record in records],
            [EmbeddingStatus.SUCCESS, EmbeddingStatus.SUCCESS],
        )
        self.assertEqual(
            [record.actual_timestamps_ms for record in records],
            [(3_000,), (8_700,)],
        )

    def test_shot_embedding_from_real_clip_embeddings(self) -> None:
        clip_vectors, clip_records = self._get_clip_result()
        service = ShotEmbeddingService(run_id="real-shot-embedding-stage-test")

        shot_vectors, shot_records = service.aggregate_shots_to_matrix(
            shots=[self.shot],
            clip_records=clip_records,
            clip_vectors=clip_vectors,
        )

        self.assertEqual(shot_vectors.shape, (1, EMBEDDING_DIMENSION))
        self.assertEqual(shot_vectors.dtype, np.float32)
        self.assertAlmostEqual(
            float(np.linalg.norm(shot_vectors[0])),
            1.0,
            places=5,
        )
        self.assertEqual(len(shot_records), 1)
        self.assertEqual(shot_records[0].status, EmbeddingStatus.SUCCESS)
        self.assertEqual(shot_records[0].entity_id, SHOT_ID)
        self.assertEqual(len(shot_records[0].source_embedding_ids), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
