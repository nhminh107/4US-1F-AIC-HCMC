from BackEnd.app.embedding.BaseEmbedding import BaseEmbedder
from PIL import Image
from abc import ABC
from BackEnd.app.contracts.pipeline import FrameMetadata
import numpy as np
from BackEnd.app.embedding import CONFIG as cf

class ImageEmbedder(BaseEmbedder):
    def get_real_data(self, data: FrameMetadata): 
        img = Image.open(data.frame_path)
        return img.convert("RGB")

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
            img = Image.open(item.frame_path).convert("RGB")
            imgs.append(img)

        return imgs

    def preprocess_batch(self, batch_data):
        return batch_data

    def encode_batch(self, batch_img):
        embeddings = self.model.encode(
            batch_img, 
            batch_size = cf.batch_size, 
            convert_to_numpy = True, 
            normaize_embeddings=True, 

        )
        return embeddings.astype(np.float32)
        