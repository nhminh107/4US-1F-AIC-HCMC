from abc import ABC, abstractmethod
import numpy as np 
import CONFIG as cf 
from sentence_transformers import SentenceTransformer, util
class BaseEmbedder(ABC): 
    def __init__(self):
        self.model = SentenceTransformer(cf.CLIP_MODEL, device=cf.device)
    def embed(self, data_record) -> np.ndarray: 
        """Đây là pipeline embedding chung cho 3 loại: Image, Clip, Shot. Gọi hàm này là sử dụng được"""

        data = self.get_real_data(data_record)
        data = self.preprocess(data)
        data = self.encode(data)

    @abstractmethod 
    def get_real_data(self, data): 
        """Data đầu vào có thể là 1 Image, Clip, Shot. Vì vậy cần hàm này để lấy data real"""
        pass

    @abstractmethod
    def preprocess(self, data): 
        """Tiền xử lí sau khi lấy data real"""
        pass

    @abstractmethod
    def encode(self, data): 
        """Tiến hành embedding"""
        pass 
