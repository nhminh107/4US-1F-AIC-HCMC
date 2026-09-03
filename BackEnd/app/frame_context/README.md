# FrameContext V1

This module creates one deterministic text document per canonical frame from
existing caption, OCR, and object-detection evidence. It does not run any ML
model and does not replace the source evidence stored in PostgreSQL.

```bash
python -m BackEnd.app.frame_context.cli \
  --build-id aic-context-v1 \
  --video-id L01_V001
```

The output directory contains `frame_context_v1.parquet` and `manifest.json`.
An existing build directory is never overwritten.

Both caption schemas currently found in this project are supported: frame-level
captions from the generic ORM schema and shot-only captions from the live import
schema. Shot captions are projected to frames by `shot_id` or timestamp.
