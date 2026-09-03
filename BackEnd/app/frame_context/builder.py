"""Pure FrameContext construction logic."""

from __future__ import annotations

from collections import Counter
import re

from BackEnd.app.frame_context.contracts import FrameContextRecord, FrameEvidence


_WHITESPACE = re.compile(r"\s+")


def _clean_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _unique_texts(values: tuple[str, ...], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _object_summary(labels: tuple[str, ...], max_labels: int) -> str:
    cleaned = [_clean_text(label) for label in labels]
    counts = Counter(label for label in cleaned if label)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
    return "; ".join(f"{label} x{count}" for label, count in ranked[:max_labels])


def build_frame_context(
    evidence: FrameEvidence,
    *,
    max_captions: int = 2,
    max_ocr_items: int = 20,
    max_object_labels: int = 20,
) -> FrameContextRecord:
    """Create one deterministic FrameContext record without changing source evidence."""

    if min(max_captions, max_ocr_items, max_object_labels) <= 0:
        raise ValueError("FrameContext limits must be positive.")

    caption_text = " ".join(_unique_texts(evidence.captions, max_captions))
    ocr_text = "; ".join(_unique_texts(evidence.ocr_texts, max_ocr_items))
    object_text = _object_summary(evidence.object_labels, max_object_labels)

    sections = []
    if caption_text:
        sections.append(f"[CAPTION]\n{caption_text}")
    if ocr_text:
        sections.append(f"[OCR]\n{ocr_text}")
    if object_text:
        sections.append(f"[OBJECTS]\n{object_text}")

    return FrameContextRecord(
        frame_id=evidence.frame_id,
        video_id=evidence.video_id,
        frame_idx=evidence.frame_idx,
        timestamp_ms=evidence.timestamp_ms,
        context_text="\n\n".join(sections),
        caption_text=caption_text,
        ocr_text=ocr_text,
        object_text=object_text,
    )
