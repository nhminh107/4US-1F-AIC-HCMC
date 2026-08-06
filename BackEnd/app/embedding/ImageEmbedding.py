from BackEnd.app.embedding import BaseEmbedding
from PIL import Image
from abc import ABC
from BackEnd.app.contracts.pipeline import FrameMetadata
import numpy as np

class ImageEmbedding(BaseEmbedding):
    def get_real_data(self, data: FrameMetadata): 
        img = Image.open(data.frame_path)
        return img.convert("RGB")

    def preprocess(self, data): 
        return data

    def encode(self, img):
        embedding = self.model.encode(
            img,
            convert_to_numpy = True, 
            normalize_embedding=True
        )

        return embedding.astype(np.float32)
