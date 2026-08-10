from BackEnd.app.embedding.BaseEmbedding import BaseEmbedder
from BackEnd.app.embedding.ImageEmbedding import ImageEmbedder
from BackEnd.app.contracts.pipeline import FrameMetadata
from pathlib import Path
from PIL import Image
embedder = ImageEmbedder()

def create_frame_metadata(frame_path):
    obj = FrameMetadata(
        frame_path=str(frame_path),
        frame_id="1", 
        shot_id="1", 
        video_id="1", 
        fps = 30.0, 
        frame_idx=int, 
        timestamp_ms=1
    )

    return obj

def create_batch_imgs(batchsize = 32): 
    p = Path.cwd()
    folder_path = p / "data" / "keyframes" / "L21_V001"

    batch_imgs = []
    for i in range(32): 
        idx = i + 1
        img_path = folder_path / f"{idx:03d}.jpg"

        obj = create_frame_metadata(img_path)
        batch_imgs.append(obj)

    return batch_imgs

def embedding(): 
    batch_imgs = create_batch_imgs()
    embed_vec = embedder.embed_batch(batch_imgs)
    return embed_vec

print(embedding())




