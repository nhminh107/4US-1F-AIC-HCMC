# Dense text indexes

This module builds separate normalized FAISS `IndexFlatIP` bundles for
FrameContext and ASR segments. The real encoder is loaded only when the CLI is
run; tests use a small fake encoder.

Build a Context index:

```bash
python -m BackEnd.app.text_embedding.cli context \
  --build-id aic-context-v1 \
  --context-artifact artifacts/context/aic-context-v1 \
  --model-id BAAI/bge-m3 \
  --model-revision YOUR_PINNED_REVISION
```

Build an ASR segment index:

```bash
python -m BackEnd.app.text_embedding.cli asr \
  --build-id aic-asr-v1 \
  --model-id BAAI/bge-m3 \
  --model-revision YOUR_PINNED_REVISION
```

Each output contains `index.faiss`, `mapping.parquet`, and `manifest.json`.
Context and ASR are deliberately kept in separate indexes because a Context
result maps to a frame while an ASR result maps to a time segment.
