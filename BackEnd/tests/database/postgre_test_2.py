"""Integration tests for PostgreSQL read operations.

The test uses ``DATABASE_URL`` from the project ``.env`` file and intentionally
keeps its uniquely named records in PostgreSQL for manual inspection.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import text

from BackEnd.app.contracts.pipeline import (
    ClipWindowMetadata,
    FrameMetadata,
    ShotMetadata,
)
from BackEnd.app.database.models import ClipWindow
from BackEnd.app.database.postgre_db import PostgreManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"


class PostgreGetIntegrationTests(unittest.TestCase):
    """Insert related records and verify every PostgreSQL get method."""

    manager: PostgreManager

    @classmethod
    def setUpClass(cls) -> None:
        if not DOTENV_PATH.is_file():
            raise RuntimeError(f"Required environment file not found: {DOTENV_PATH}")

        load_dotenv(DOTENV_PATH, override=True)
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError(
                "DATABASE_URL is not configured in the project .env file."
            )

        cls.manager = PostgreManager()
        with cls.manager.engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "manager"):
            cls.manager.engine.dispose()

    def test_add_records_and_check_all_get_methods(self) -> None:
        suffix = uuid4().hex[:10]
        video_id = f"v{suffix}"
        first_shot_id = f"s0{suffix}"
        second_shot_id = f"s1{suffix}"
        first_frame_id = f"f0{suffix}"
        second_frame_id = f"f1{suffix}"
        other_frame_id = f"f2{suffix}"
        first_clip_id = f"c0{suffix}"
        second_clip_id = f"c1{suffix}"
        other_clip_id = f"c2{suffix}"

        self.manager.add_video(
            video_id=video_id,
            video_path=f"tests/{video_id}.mp4",
            title="PostgreSQL get integration test",
            duration_ms=10_000,
        )
        self.manager.add_shot(
            shot_id=first_shot_id,
            video_id=video_id,
            shot_index=0,
            start_ms=0,
            end_ms=5_000,
            start_frame_idx=0,
            end_frame_idx=49,
        )
        self.manager.add_shot(
            shot_id=second_shot_id,
            video_id=video_id,
            shot_index=1,
            start_ms=5_000,
            end_ms=10_000,
            start_frame_idx=50,
            end_frame_idx=99,
        )

        # Insert frames out of order to verify ordering in the get method.
        self.manager.add_frame(
            frame_id=second_frame_id,
            video_id=video_id,
            shot_id=first_shot_id,
            timestamp_ms=1_000,
            fps=30.0,
            frame_idx=30,
            frame_role="keyframe",
            source="extracted",
            frame_path=f"tests/{second_frame_id}.jpg",
            width=1920,
            height=1080,
        )
        self.manager.add_frame(
            frame_id=first_frame_id,
            video_id=video_id,
            shot_id=first_shot_id,
            timestamp_ms=100,
            fps=30.0,
            frame_idx=3,
            frame_role="keyframe",
            source="extracted",
            frame_path=f"tests/{first_frame_id}.jpg",
            width=1920,
            height=1080,
        )
        self.manager.add_frame(
            frame_id=other_frame_id,
            video_id=video_id,
            shot_id=second_shot_id,
            timestamp_ms=6_000,
            fps=30.0,
            frame_idx=60,
            frame_role="keyframe",
            source="extracted",
            frame_path=f"tests/{other_frame_id}.jpg",
            width=1920,
            height=1080,
        )

        with self.manager.session_factory.begin() as session:
            session.add_all(
                [
                    ClipWindow(
                        clip_id=second_clip_id,
                        shot_id=first_shot_id,
                        start_ms=2_000,
                        end_ms=3_000,
                        start_frame_idx=20,
                        end_frame_idx=29,
                        sampling_fps=5.0,
                        clip_path=f"tests/{second_clip_id}.mp4",
                    ),
                    ClipWindow(
                        clip_id=first_clip_id,
                        shot_id=first_shot_id,
                        start_ms=500,
                        end_ms=1_500,
                        start_frame_idx=5,
                        end_frame_idx=14,
                        sampling_fps=5.0,
                        clip_path=f"tests/{first_clip_id}.mp4",
                    ),
                    ClipWindow(
                        clip_id=other_clip_id,
                        shot_id=second_shot_id,
                        start_ms=5_500,
                        end_ms=6_500,
                        start_frame_idx=55,
                        end_frame_idx=64,
                        sampling_fps=5.0,
                        clip_path=f"tests/{other_clip_id}.mp4",
                    ),
                ]
            )

        frame = self.manager.get_frame_record_by_frame_id(first_frame_id)
        frames = self.manager.get_list_frame_in_shot(first_shot_id)
        shots = self.manager.get_list_shot_in_video(video_id)
        clips = self.manager.get_list_clip_in_shot(first_shot_id)

        self.assertIsInstance(frame, FrameMetadata)
        self.assertEqual(frame.frame_id, first_frame_id)
        self.assertEqual(frame.frame_path, Path(f"tests/{first_frame_id}.jpg"))

        self.assertEqual(
            [item.frame_id for item in frames],
            [first_frame_id, second_frame_id],
        )
        self.assertTrue(all(isinstance(item, FrameMetadata) for item in frames))

        self.assertEqual(
            [item.shot_id for item in shots],
            [first_shot_id, second_shot_id],
        )
        self.assertTrue(all(isinstance(item, ShotMetadata) for item in shots))

        self.assertEqual(
            [item.clip_id for item in clips],
            [first_clip_id, second_clip_id],
        )
        self.assertTrue(all(isinstance(item, ClipWindowMetadata) for item in clips))
        self.assertEqual(clips[0].clip_path, Path(f"tests/{first_clip_id}.mp4"))


if __name__ == "__main__":
    unittest.main()
