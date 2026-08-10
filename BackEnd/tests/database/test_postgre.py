"""Integration tests for PostgreSQL persistence operations.

These tests use ``DATABASE_URL`` from the project ``.env`` file. Every test
creates records with random IDs and intentionally keeps them in the database
for manual inspection.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from BackEnd.app.database.models import (
    Frame,
    FrameEmbeddingRecord,
    OCR,
    Shot,
    Video,
)
from BackEnd.app.database.postgre_db import PostgreManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
REQUIRED_TABLES = {
    "caption",
    "frame",
    "frameembeddingrecord",
    "ocr",
    "shot",
    "transcriptsegment",
    "video",
}


class PostgreManagerIntegrationTests(unittest.TestCase):
    """Exercise add methods against the configured PostgreSQL database."""

    manager: PostgreManager

    @classmethod
    def setUpClass(cls) -> None:
        if not DOTENV_PATH.is_file():
            raise RuntimeError(f"Required environment file not found: {DOTENV_PATH}")

        load_dotenv(DOTENV_PATH, override=True)
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("DATABASE_URL is not configured in the project .env file.")

        cls.manager = PostgreManager()
        with cls.manager.engine.connect() as connection:
            cls.database_name, cls.database_user = connection.execute(
                text("SELECT current_database(), current_user")
            ).one()
            existing_tables = set(
                connection.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public'"
                    )
                ).scalars()
            )

        missing_tables = REQUIRED_TABLES - existing_tables
        if missing_tables:
            missing_names = ", ".join(sorted(missing_tables))
            raise RuntimeError(f"Required database tables are missing: {missing_names}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.manager.engine.dispose()

    def setUp(self) -> None:
        unique_suffix = uuid4().hex[:10]
        self.video_id = f"tv{unique_suffix}"
        self.shot_id = f"ts{unique_suffix}"
        self.frame_id = f"tf{unique_suffix}"
        self.faiss_id = int(uuid4().hex[:12], 16)

    def add_test_video(self) -> Video:
        """Persist the video shared by dependent-record tests."""

        return self.manager.add_video(
            video_id=self.video_id,
            video_path=f"tests/{self.video_id}.mp4",
            title="PostgreSQL integration test",
            duration_ms=10_000,
        )

    def add_test_shot(self) -> Shot:
        """Persist the video and shot shared by frame tests."""

        self.add_test_video()
        return self.manager.add_shot(
            shot_id=self.shot_id,
            video_id=self.video_id,
            shot_index=0,
            start_ms=0,
            end_ms=5_000,
            start_frame_idx=0,
            end_frame_idx=149,
        )

    def add_test_frame(self) -> Frame:
        """Persist the video, shot, and frame shared by child-record tests."""

        self.add_test_shot()
        return self.manager.add_frame(
            frame_id=self.frame_id,
            video_id=self.video_id,
            shot_id=self.shot_id,
            timestamp_ms=1_000,
            fps=30.0,
            frame_idx=30,
            source="extracted",
            frame_path=f"tests/{self.frame_id}.jpg",
            width=1920,
            height=1080,
        )

    def test_database_url_connects_to_postgresql(self) -> None:
        with self.manager.engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar_one()
            current_database = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()

        self.assertEqual(result, 1)
        self.assertEqual(current_database, self.database_name)
        self.assertEqual(self.manager.engine.url.get_backend_name(), "postgresql")

    def test_missing_database_url_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                PostgreManager()

    def test_add_video_writes_and_reads_real_record(self) -> None:
        video = self.add_test_video()

        with Session(self.manager.engine) as session:
            persisted_video = session.get(Video, self.video_id)

        self.assertEqual(video.video_id, self.video_id)
        self.assertIsNotNone(persisted_video)
        self.assertEqual(persisted_video.video_path, f"tests/{self.video_id}.mp4")
        self.assertEqual(persisted_video.duration_ms, 10_000)
        print(f"Persisted video: {self.video_id}")

    def test_add_shot_and_frame_write_relationships(self) -> None:
        frame = self.add_test_frame()

        with Session(self.manager.engine) as session:
            persisted_frame = session.get(Frame, self.frame_id)
            persisted_shot = session.get(Shot, self.shot_id)

        self.assertEqual(frame.frame_id, self.frame_id)
        self.assertIsNotNone(persisted_frame)
        self.assertIsNotNone(persisted_shot)
        self.assertEqual(persisted_frame.shot_id, self.shot_id)
        self.assertEqual(persisted_shot.video_id, self.video_id)
        print(
            f"Persisted video/shot/frame: "
            f"{self.video_id}/{self.shot_id}/{self.frame_id}"
        )

    def test_add_ocr_and_embedding_write_child_records(self) -> None:
        self.add_test_frame()
        ocr = self.manager.add_ocr(
            frame_id=self.frame_id,
            n=0,
            text="Integration test text",
            x_min=0.1,
            x_max=0.7,
            y_min=0.2,
            y_max=0.4,
            language="en",
        )
        embedding = self.manager.add_frame_embedding_record(
            faiss_id=self.faiss_id,
            index_version=1,
            frame_id=self.frame_id,
            model_name="test-embedding-model",
        )

        with Session(self.manager.engine) as session:
            persisted_ocr = session.get(OCR, (self.frame_id, 0))
            persisted_embedding = session.get(
                FrameEmbeddingRecord, (self.faiss_id, 1)
            )

        self.assertEqual(ocr.text, "Integration test text")
        self.assertEqual(embedding.faiss_id, self.faiss_id)
        self.assertIsNotNone(persisted_ocr)
        self.assertIsNotNone(persisted_embedding)
        print(
            f"Persisted OCR/embedding for frame: "
            f"{self.frame_id}, faiss_id={self.faiss_id}"
        )

    def test_duplicate_video_fails_without_writing_second_record(self) -> None:
        self.add_test_video()

        with self.assertRaises(IntegrityError):
            self.manager.add_video(
                video_id=self.video_id,
                video_path="tests/duplicate.mp4",
            )

        with Session(self.manager.engine) as session:
            matching_rows = session.scalar(
                select(func.count())
                .select_from(Video)
                .where(Video.video_id == self.video_id)
            )
            persisted_video = session.get(Video, self.video_id)

        self.assertEqual(matching_rows, 1)
        self.assertEqual(persisted_video.video_path, f"tests/{self.video_id}.mp4")
        print(f"Persisted one video after duplicate rejection: {self.video_id}")

    def test_add_shot_rejects_missing_video_without_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.manager.add_shot(
                shot_id=self.shot_id,
                video_id=self.video_id,
                shot_index=0,
                start_ms=0,
                end_ms=1_000,
            )

        with Session(self.manager.engine) as session:
            persisted_shot = session.get(Shot, self.shot_id)

        self.assertIsNone(persisted_shot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
