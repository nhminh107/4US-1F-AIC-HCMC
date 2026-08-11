"""Persistence and recovery tests for the FAISS manager."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.database.faiss_db import FAISS_Manager


def _frame(frame_id: str) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id="L21_V005",
        shot_id="L21_V005_S000",
        timestamp_ms=0,
        fps=30.0,
        frame_idx=0,
        source="extracted",
        frame_path=Path("data/frame.jpg"),
    )


def test_manager_loads_existing_index_and_rolls_back_new_ids(tmp_path: Path) -> None:
    first_manager = FAISS_Manager(2, 2, 2, data_path=tmp_path)
    first_mappings, _, _ = first_manager.add_and_save(
        imgs=np.asarray([[1.0, 0.0]], dtype=np.float32),
        imgs_model=[_frame("L21_V005_E001")],
    )

    reloaded_manager = FAISS_Manager(2, 2, 2, data_path=tmp_path)
    assert reloaded_manager.frame_idx.ntotal == 1

    second_mappings, _, _ = reloaded_manager.add_and_save(
        imgs=np.asarray([[0.0, 1.0]], dtype=np.float32),
        imgs_model=[_frame("L21_V005_E002")],
    )
    assert first_mappings[0].faiss_id == 1
    assert second_mappings[0].faiss_id == 2

    reloaded_manager.rollback(frame_mappings=second_mappings)

    final_manager = FAISS_Manager(2, 2, 2, data_path=tmp_path)
    assert final_manager.frame_idx.ntotal == 1
    counters = json.loads((tmp_path / "faiss_index.json").read_text())
    assert counters["image_idx"] == 1


def test_failed_save_restores_the_persisted_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = FAISS_Manager(2, 2, 2, data_path=tmp_path)
    manager.add_and_save(
        imgs=np.asarray([[1.0, 0.0]], dtype=np.float32),
        imgs_model=[_frame("L21_V005_E001")],
    )

    def fail_counter_save() -> None:
        raise OSError("counter write failed")

    monkeypatch.setattr(manager, "_save_id_counters", fail_counter_save)
    with pytest.raises(RuntimeError, match="compensated index"):
        manager.add_and_save(
            imgs=np.asarray([[0.0, 1.0]], dtype=np.float32),
            imgs_model=[_frame("L21_V005_E002")],
        )

    reloaded = FAISS_Manager(2, 2, 2, data_path=tmp_path)
    assert reloaded.frame_idx.ntotal == 1
