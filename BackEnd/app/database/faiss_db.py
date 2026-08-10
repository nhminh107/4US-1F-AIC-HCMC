import json
import os
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from BackEnd.app.contracts.pipeline import (
    ClipEmbeddingMapping,
    ClipWindowMetadata,
    FrameEmbeddingMapping,
    FrameMetadata,
    ShotEmbeddingMapping,
    ShotMetadata,
)


class FAISS_Manager:
    def __init__(
        self,
        img_dim,
        clip_dim,
        shot_dim,
        version=0,
        model_name="clip-ViT-B-32",
        model_version="0",
        data_path: str | Path | None = None,
        load_existing: bool = True,
    ):
        self.datapath = Path(data_path or Path(__file__).resolve().parent)
        self.datapath.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.model_name = model_name
        self.model_version = model_version
        self.frame_idx = self._load_or_create_index(
            "frame.faiss", img_dim, load_existing
        )
        self.clip_idx = self._load_or_create_index(
            "clip.faiss", clip_dim, load_existing
        )
        self.shot_idx = self._load_or_create_index(
            "shot.faiss", shot_dim, load_existing
        )

    def _load_or_create_index(
        self,
        filename: str,
        expected_dim: int,
        load_existing: bool,
    ):
        path = self.datapath / filename
        if load_existing and path.is_file():
            index = faiss.read_index(str(path))
            if index.d != expected_dim:
                raise ValueError(
                    f"Existing {filename} dimension is {index.d}, expected {expected_dim}."
                )
            if not isinstance(index, faiss.IndexIDMap2):
                raise ValueError(f"Existing {filename} must be a FAISS IndexIDMap2.")
            return index

        return faiss.IndexIDMap2(faiss.IndexFlatIP(expected_dim))

    def __prepare_embeddings(self, embeddings, expected_dim):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embeddings must have shape (batch_size, {expected_dim})"
            )
        return embeddings

    @staticmethod
    def _index_ids(index) -> np.ndarray:
        return faiss.vector_to_array(index.id_map).astype(np.int64, copy=False)

    def __get_ids(self, index, n):
        current_ids = self._index_ids(index)
        first_idx = int(current_ids.max()) + 1 if current_ids.size else 1
        return np.arange(first_idx, first_idx + n, dtype=np.int64)

    def add_imgs(self, imgs: list[np.ndarray], imgs_model: list[FrameMetadata]):
        if len(imgs) != len(imgs_model):
            raise ValueError("imgs and imgs_model must have the same length")

        imgs = self.__prepare_embeddings(imgs, self.frame_idx.d)
        ids = self.__get_ids(self.frame_idx, len(imgs))

        self.frame_idx.add_with_ids(imgs, ids)

        frame_embedding_records = []
        for count, faiss_id in enumerate(ids):
            record = FrameEmbeddingMapping(
                faiss_id=int(faiss_id),
                index_version=self.version,
                frame_id=imgs_model[count].frame_id,
                model_name=self.model_name,
            )
            frame_embedding_records.append(record)
        return frame_embedding_records

    def add_clips(self, clips: list[np.ndarray], clips_model: list[ClipWindowMetadata]):
        if len(clips) != len(clips_model):
            raise ValueError("clips and clips_model must have the same length")

        clips = self.__prepare_embeddings(clips, self.clip_idx.d)
        ids = self.__get_ids(self.clip_idx, len(clips))

        self.clip_idx.add_with_ids(clips, ids)

        clip_embedding_records = []
        for count, faiss_id in enumerate(ids):
            record = ClipEmbeddingMapping(
                faiss_id=int(faiss_id),
                index_version=self.version,
                clip_id=clips_model[count].clip_id,
                model_name=self.model_name,
            )
            clip_embedding_records.append(record)
        return clip_embedding_records

    def add_shots(self, shots: list[np.ndarray], shots_model: list[ShotMetadata]):
        if len(shots) != len(shots_model):
            raise ValueError("shots and shots_model must have the same length")

        shots = self.__prepare_embeddings(shots, self.shot_idx.d)
        ids = self.__get_ids(self.shot_idx, len(shots))

        self.shot_idx.add_with_ids(shots, ids)

        shot_embedding_records = []
        for count, faiss_id in enumerate(ids):
            record = ShotEmbeddingMapping(
                faiss_id=int(faiss_id),
                index_version=self.version,
                shot_id=shots_model[count].shot_id,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            shot_embedding_records.append(record)
        return shot_embedding_records

    def save(self, index_types: set[str] | None = None):
        """Atomically persist selected indexes and synchronize ID counters."""

        selected = index_types or {"frame", "clip", "shot"}
        indexes = {
            "frame": (self.frame_idx, self.datapath / "frame.faiss"),
            "clip": (self.clip_idx, self.datapath / "clip.faiss"),
            "shot": (self.shot_idx, self.datapath / "shot.faiss"),
        }
        unknown = selected - set(indexes)
        if unknown:
            raise ValueError(f"Unknown FAISS index types: {sorted(unknown)}")

        for index_type in selected:
            index, path = indexes[index_type]
            temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                faiss.write_index(index, str(temporary_path))
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)

        self._save_id_counters()

    def _save_id_counters(self) -> None:
        counter_path = self.datapath / "faiss_index.json"
        temporary_path = counter_path.with_name(
            f".{counter_path.name}.{uuid4().hex}.tmp"
        )
        counters = {
            "image_idx": self._max_id(self.frame_idx),
            "clip_idx": self._max_id(self.clip_idx),
            "shot_idx": self._max_id(self.shot_idx),
        }
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(counters, file, indent=4)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, counter_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def _max_id(cls, index) -> int:
        ids = cls._index_ids(index)
        return int(ids.max()) if ids.size else 0

    def validate_ids(self, index_type: str, faiss_ids: list[int]) -> None:
        """Ensure persisted PostgreSQL mappings still exist in the FAISS index."""

        indexes = {
            "frame": self.frame_idx,
            "clip": self.clip_idx,
            "shot": self.shot_idx,
        }
        if index_type not in indexes:
            raise ValueError(f"Unknown FAISS index type: {index_type}")
        available_ids = set(self._index_ids(indexes[index_type]).tolist())
        missing_ids = sorted(set(faiss_ids) - available_ids)
        if missing_ids:
            raise RuntimeError(
                f"PostgreSQL references missing {index_type} FAISS IDs: "
                f"{missing_ids[:10]}."
            )

    @staticmethod
    def _remove_mappings(index, mappings) -> None:
        ids = np.asarray([mapping.faiss_id for mapping in mappings], dtype=np.int64)
        if ids.size:
            index.remove_ids(ids)

    def rollback(
        self,
        *,
        frame_mappings=None,
        clip_mappings=None,
        shot_mappings=None,
    ) -> None:
        """Remove newly-added mappings and persist the compensated indexes."""

        selected: set[str] = set()
        if frame_mappings:
            self._remove_mappings(self.frame_idx, frame_mappings)
            selected.add("frame")
        if clip_mappings:
            self._remove_mappings(self.clip_idx, clip_mappings)
            selected.add("clip")
        if shot_mappings:
            self._remove_mappings(self.shot_idx, shot_mappings)
            selected.add("shot")
        if selected:
            self.save(selected)

    def add_and_save(
        self,
        imgs=None,
        imgs_model=None,
        clips=None,
        clips_model=None,
        shots=None,
        shots_model=None,
    ):
        frame_records = []
        clip_records = []
        shot_records = []
        selected: set[str] = set()

        if imgs is not None and imgs_model is not None:
            frame_records = self.add_imgs(imgs, imgs_model)
            selected.add("frame")
        if clips is not None and clips_model is not None:
            clip_records = self.add_clips(clips, clips_model)
            selected.add("clip")
        if shots is not None and shots_model is not None:
            shot_records = self.add_shots(shots, shots_model)
            selected.add("shot")

        try:
            if selected:
                self.save(selected)
        except Exception:
            self._remove_mappings(self.frame_idx, frame_records)
            self._remove_mappings(self.clip_idx, clip_records)
            self._remove_mappings(self.shot_idx, shot_records)
            try:
                if selected:
                    self.save(selected)
            except Exception as rollback_error:
                raise RuntimeError(
                    "FAISS save failed and the compensated index could not be persisted."
                ) from rollback_error
            raise
        return frame_records, clip_records, shot_records


if __name__ == "__main__":
    manager = FAISS_Manager(4, 4, 4, model_version="test")

    imgs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    clips = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    shots = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    imgs_model = [
        FrameMetadata(
            frame_id="test-frame-1",
            video_id="test-video-1",
            shot_id="test-shot-1",
            timestamp_ms=0,
            fps=30.0,
            frame_idx=0,
            source="extracted",
        )
    ]
    clips_model = [
        ClipWindowMetadata(
            clip_id="test-clip-1",
            shot_id="test-shot-1",
            start_ms=0,
            end_ms=1000,
        )
    ]
    shots_model = [
        ShotMetadata(
            shot_id="test-shot-1",
            video_id="test-video-1",
            shot_index=0,
            start_ms=0,
            end_ms=1000,
        )
    ]

    frame_records, clip_records, shot_records = manager.add_and_save(
        imgs=imgs,
        imgs_model=imgs_model,
        clips=clips,
        clips_model=clips_model,
        shots=shots,
        shots_model=shots_model,
    )

    print("Frame records:", frame_records)
    print("Clip records:", clip_records)
    print("Shot records:", shot_records)
    print("Frame total:", manager.frame_idx.ntotal)
    print("Clip total:", manager.clip_idx.ntotal)
    print("Shot total:", manager.shot_idx.ntotal)
