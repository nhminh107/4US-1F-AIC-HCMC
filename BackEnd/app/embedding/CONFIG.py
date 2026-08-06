from enum import Enum
import torch

class EntityType(str, Enum):
    FRAME = "frame"
    CLIP = "clip"
    SHOT = "shot"


class EmbeddingStatus(str, Enum):
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    MODEL_FAILED = "model_failed"
    INVALID_VECTOR = "invalid_vector"

CLIP_MODEL = "clip-ViT-B-32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
batch_size = 64