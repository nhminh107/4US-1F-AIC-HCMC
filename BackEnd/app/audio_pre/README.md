# Audio Preprocessing

Shot-level audio preprocessing for the ASR stage.

The module accepts `VideoMetadata` plus `list[ShotMetadata]`, extracts the
full-video audio once, cuts one WAV per shot using exact shot boundaries,
normalizes every successful shot to 16 kHz mono PCM16 with peak target `1.0`,
runs WebRTC VAD, and returns `list[AudioSegment]`.

It does not transcribe, detect language, diarize speakers, or alter visual shot
boundaries.

## Programmatic Usage

```python
from pathlib import Path

from BackEnd.app.audio_pre import preprocess_video
from BackEnd.app.contracts.pipeline import ShotMetadata, VideoMetadata

segments = preprocess_video(
    VideoMetadata(video_id="video001", video_path=Path("video001.mp4")),
    [ShotMetadata("shot-1", "video001", 0, 0, 5200)],
    Path("output"),
    language_hint="vi",
)
```

## CLI

```bash
python -m BackEnd.app.audio_pre.run_preprocessing \
  --video video001.mp4 \
  --shots shots.json \
  --output-dir output \
  --language vi
```

Outputs are written under `output/{video_id}/`, including normalized shot WAVs
and `audio_segments.json`. The temporary `{video_id}_raw.wav` is deleted after
the segmentation phase.
