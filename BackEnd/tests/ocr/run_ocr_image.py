"""Run OCR on one image and print the pipeline contract results.

Example:
    python -m BackEnd.tests.ocr.run_ocr_image data/keyframes/L21_V001/001.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.ocr import run_ocr


def main(img_path: Path) -> None:
    """Create one frame contract, call OCR, and print its results."""

    video_id = img_path.parent.name
    frame_number = int(img_path.stem) if img_path.stem.isdigit() else 0
    frame = FrameMetadata(
        frame_id=f"{video_id}_{img_path.stem}",
        video_id=video_id,
        shot_id=f"{video_id}_S000",
        timestamp_ms=0,
        fps=1.0,
        frame_idx=0,
        source="official",
        n=frame_number,
        frame_path=img_path,
    )

    results = run_ocr(frame)
    if not results:
        print("No text detected.")
        return

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on one image.")
    parser.add_argument("img_path", type=Path, help="Path to the input image.")
    arguments = parser.parse_args()
    main(arguments.img_path)
