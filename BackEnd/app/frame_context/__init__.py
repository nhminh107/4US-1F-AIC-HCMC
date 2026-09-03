"""Build deterministic frame-level text evidence for semantic retrieval."""

from BackEnd.app.frame_context.builder import build_frame_context
from BackEnd.app.frame_context.contracts import FrameContextRecord, FrameEvidence

__all__ = ["FrameContextRecord", "FrameEvidence", "build_frame_context"]
