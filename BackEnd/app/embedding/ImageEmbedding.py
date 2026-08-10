from BackEnd.app.embedding.BaseEmbedding import BaseEmbedder
from PIL import Image
from abc import ABC
from pathlib import Path
from BackEnd.app.contracts.pipeline import FrameMetadata
import numpy as np
from BackEnd import CONFIG as cf

class ImageEmbedder(BaseEmbedder):
    def get_real_data(self, data: FrameMetadata): 
        path = self._resolve_frame_path(data)
        with Image.open(path) as image:
            return image.convert("RGB")

    def preprocess(self, data): 
        return data

    def encode(self, img):
        embedding = self.model.encode(
            img,
            convert_to_numpy = True, 
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)

    def get_real_data_list(self, batch_data: list[FrameMetadata]): 

        imgs = []

        for item in batch_data: 
            path = self._resolve_frame_path(item)
            with Image.open(path) as image:
                imgs.append(image.convert("RGB"))

        return imgs

    def preprocess_batch(self, batch_data):
        return batch_data

    def encode_batch(self, batch_img):
        embeddings = self.model.encode(
            batch_img, 
            batch_size = cf.batch_size, 
            convert_to_numpy = True, 
            normalize_embeddings=True,

        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _resolve_frame_path(data: FrameMetadata):
        if data.frame_path is None:
            raise ValueError(f"Frame '{data.frame_id}' does not have frame_path.")
        path = Path(data.frame_path)
        if not path.is_absolute():
            path = cf.PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Frame image does not exist for '{data.frame_id}': {path}."
            )
        return path
