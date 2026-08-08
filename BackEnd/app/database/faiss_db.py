import json
from pathlib import Path

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
    ):
        self.frame = faiss.IndexFlatIP(img_dim)
        self.clip = faiss.IndexFlatIP(clip_dim)
        self.shot = faiss.IndexFlatIP(shot_dim)

        self.frame_idx = faiss.IndexIDMap2(self.frame)
        self.clip_idx = faiss.IndexIDMap2(self.clip)
        self.shot_idx = faiss.IndexIDMap2(self.shot)

        self.datapath = Path(__file__).resolve().parent
        self.version = version
        self.model_name = model_name
        self.model_version = model_version

    def __get_idx_field(self, index_type: int):
        """Type: {0: image_idx, 1: clip_idx, 2: shot_idx}"""
        if index_type == 0:
            return "image_idx"
        if index_type == 1:
            return "clip_idx"
        if index_type == 2:
            return "shot_idx"
        raise ValueError("Index type does not exist")

    def __get_idx_json(self, index_type: int):
        json_path = self.datapath / "faiss_index.json"
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data[self.__get_idx_field(index_type)]

    def __buff_idx_json(self, index_type: int, n=1):
        """Use after add_with_ids."""
        json_path = self.datapath / "faiss_index.json"
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        field = self.__get_idx_field(index_type)
        data[field] += n
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def __prepare_embeddings(self, embeddings, expected_dim):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embeddings must have shape (batch_size, {expected_dim})"
            )
        return embeddings

    def __get_ids(self, index_type, n):
        first_idx = self.__get_idx_json(index_type) + 1
        return np.arange(first_idx, first_idx + n, dtype=np.int64)

    def add_imgs(self, imgs: list[np.ndarray], imgs_model: list[FrameMetadata]):
        if len(imgs) != len(imgs_model):
            raise ValueError("imgs and imgs_model must have the same length")

        imgs = self.__prepare_embeddings(imgs, self.frame_idx.d)
        ids = self.__get_ids(0, len(imgs))

        self.frame_idx.add_with_ids(imgs, ids)
        self.__buff_idx_json(0, len(imgs))

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
        ids = self.__get_ids(1, len(clips))

        self.clip_idx.add_with_ids(clips, ids)
        self.__buff_idx_json(1, len(clips))

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
        ids = self.__get_ids(2, len(shots))

        self.shot_idx.add_with_ids(shots, ids)
        self.__buff_idx_json(2, len(shots))

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

    def save(self):
        faiss.write_index(self.frame_idx, str(self.datapath / "frame.faiss"))
        faiss.write_index(self.clip_idx, str(self.datapath / "clip.faiss"))
        faiss.write_index(self.shot_idx, str(self.datapath / "shot.faiss"))

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

        if imgs is not None and imgs_model is not None:
            frame_records = self.add_imgs(imgs, imgs_model)
        if clips is not None and clips_model is not None:
            clip_records = self.add_clips(clips, clips_model)
        if shots is not None and shots_model is not None:
            shot_records = self.add_shots(shots, shots_model)

        self.save()
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
            frame_role="keyframe",
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
