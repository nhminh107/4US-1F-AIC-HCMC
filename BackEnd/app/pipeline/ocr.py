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
) -> list[OCRResult]:
    """Run OCR on one video's frames in bounded-memory chunks."""

    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive.")

    frames = db.get_frame_record_by_video_id(video_id)
    all_results: list[OCRResult] = []
    for start in range(0, len(frames), frame_chunk_size):
        results = run_ocr_batch(
            frames[start : start + frame_chunk_size],
            service=service,
        )
        all_results.extend(results)

    # Persist only after all chunks finish. An OOM during inference can then be
    # retried in a new worker without violating OCR's (frame_id, n) key.
    for result in all_results:
        db.add_ocr(
            frame_id=result.frame_id,
            n=result.n,
            text=result.text,
            x_min=result.x_min,
            x_max=result.x_max,
            y_min=result.y_min,
            y_max=result.y_max,
            language=result.language,
        )

    return all_results


__all__ = ["run_ocr"]
