"""Run OCR for all persisted frames of one video."""

from __future__ import annotations

from BackEnd.app.contracts.pipeline import OCRResult
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.ocr.run_ocr_pipeline import run_ocr_batch
from BackEnd.app.ocr.service import OCRService


def run_ocr(
    video_id: str,
    db: PostgreManager,
    service: OCRService,
) -> list[OCRResult]:
    """Run OCR on one video's frames and persist the results."""

    frames = db.get_frame_record_by_video_id(video_id)
    results = run_ocr_batch(frames, service=service)

    for result in results:
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

    return results


__all__ = ["run_ocr"]
