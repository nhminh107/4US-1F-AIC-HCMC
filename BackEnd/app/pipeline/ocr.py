"""Run OCR for all persisted frames of one video."""

from __future__ import annotations

from BackEnd.CONFIG import OCR_FRAME_CHUNK_SIZE
from BackEnd.app.contracts.pipeline import OCRResult
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.ocr.run_ocr_pipeline import run_ocr_batch
from BackEnd.app.ocr.service import OCRService


def run_ocr(
    video_id: str,
    db: PostgreManager,
    service: OCRService,
    *,
    frame_chunk_size: int = OCR_FRAME_CHUNK_SIZE,
    frame_source: str | None = None,
) -> list[OCRResult]:
    """Run OCR on one video's frames in bounded-memory chunks."""

    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive.")

    frames = db.get_frame_record_by_video_id(video_id)
    if frame_source is not None:
        frames = [frame for frame in frames if frame.source == frame_source]
    all_results: list[OCRResult] = []
    for start in range(0, len(frames), frame_chunk_size):
        results = run_ocr_batch(
            frames[start : start + frame_chunk_size],
            service=service,
        )
        all_results.extend(results)

    # Persist only after all chunks finish. An OOM during inference can then be
    # retried in a new worker without violating OCR's (frame_id, n) key.
    db.add_ocr_records(all_results)

    return all_results


__all__ = ["run_ocr"]
