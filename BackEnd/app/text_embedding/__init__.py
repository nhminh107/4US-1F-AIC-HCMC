"""Dense text embedding and FAISS index construction."""

from BackEnd.app.text_embedding.builder import build_text_index
from BackEnd.app.text_embedding.contracts import TextDocument, TextIndexManifest

__all__ = ["TextDocument", "TextIndexManifest", "build_text_index"]
