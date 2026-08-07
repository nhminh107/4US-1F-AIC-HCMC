"""Text query encoder for CLIP-compatible benchmarks."""

from __future__ import annotations

from BackEnd.app.embedding.models.clip_vit_b32 import ClipViTB32Adapter


def encode_clip_queries(
    texts: list[str],
    adapter: ClipViTB32Adapter | None = None,
):
    """Encode query texts using the same CLIP checkpoint as visual vectors."""

    resolved_adapter = adapter or ClipViTB32Adapter()
    return resolved_adapter.encode_texts(texts)

