# Embedding usage

## Image embedding

Use `ImageEmbedder.embed()` with a `FrameMetadata` record:

```python
from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.embedding.ImageEmbedding import ImageEmbedder

frame = FrameMetadata(
    frame_id="L21_V001-001",
    video_id="L21_V001",
    shot_id="shot-1",
    timestamp_ms=0,
    fps=30.0,
    frame_idx=0,
    source="official",
    n=1,
    frame_path=Path("data/keyframes/L21_V001/001.jpg"),
)

embedder = ImageEmbedder()
image_vector = embedder.embed(frame)

print(image_vector.shape)  # (512,)
```

## Clip embedding

`ClipEmbeddingService` requires:

- A video decoder implementing `VideoDecoder`.
- A CLIP model adapter implementing `ImageTextEmbeddingAdapter`.
- A list of `ClipRecord` objects.
- A mapping from `video_id` to `VideoAsset`.

```python
from BackEnd.app.contracts.embedding import EmbeddingStatus
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.embedding.clip.builder import build_clips
from BackEnd.app.embedding.clip.decoder import PyAVVideoDecoder
from BackEnd.app.embedding.clip.service import ClipEmbeddingService
from BackEnd.app.embedding.clip.video_repository import VideoRepository

# Your adapter must implement ImageTextEmbeddingAdapter:
# encode_images(), encode_texts(), get_dimension(), and metadata().
model_adapter = your_clip_model_adapter

shot = ShotMetadata(
    shot_id="shot-1",
    video_id="L21_V001",
    shot_index=0,
    start_ms=0,
    end_ms=10_000,
)

clips = build_clips([shot])
video_asset = VideoRepository("data/video").resolve_video("L21_V001")

clip_service = ClipEmbeddingService(
    decoder=PyAVVideoDecoder(),
    model_adapter=model_adapter,
)

clip_vectors, clip_records = clip_service.embed_clips_to_matrix(
    clips,
    {video_asset.video_id: video_asset},
)

successful_clip_records = [
    record
    for record in clip_records
    if record.status == EmbeddingStatus.SUCCESS
]

print(clip_vectors.shape)  # (number_of_successful_clips, 512)
```

To write the clip vectors and metadata as an artifact instead of returning a
matrix:

```python
clip_manifest = clip_service.embed_clips(
    clips,
    {video_asset.video_id: video_asset},
)
```

## Shot embedding

Use the output of `embed_clips_to_matrix()` as the input of
`aggregate_shots_to_matrix()`:

```python
from BackEnd.app.embedding.shot.service import ShotEmbeddingService

shot_service = ShotEmbeddingService()

shot_vectors, shot_records = shot_service.aggregate_shots_to_matrix(
    shots=[shot],
    clip_records=successful_clip_records,
    clip_vectors=clip_vectors,
)

print(shot_vectors.shape)  # (number_of_successful_shots, 512)
```

To build a shot artifact from an existing clip artifact:

```python
from pathlib import Path

clip_artifact_root = (
    Path("artifacts/embeddings")
    / clip_manifest.embedding_space_id
    / clip_manifest.run_id
)

shot_manifest = shot_service.aggregate_from_clip_artifact(
    shots=[shot],
    clip_manifest=clip_manifest,
    clip_artifact_root=clip_artifact_root,
)
```
