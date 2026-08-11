from __future__ import annotations

from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.pipeline.ocr import run_ocr


class FakeDatabase:
    def __init__(self, frames: list[FrameMetadata]) -> None:
        self.frames = frames
        self.saved: list[dict[str, object]] = []

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        assert video_id == "L23_V005"
        return self.frames

    def add_ocr(self, **values: object) -> None:
        self.saved.append(values)


class FakeOCRService:
    def __init__(self, results: list[OCRResult]) -> None:
        self.results = results
        self.received_frames: list[FrameMetadata] = []

    def process_batch(self, frames: list[FrameMetadata]) -> list[OCRResult]:
        self.received_frames = frames
        return self.results


def test_run_ocr_processes_and_persists_video_frames() -> None:
    frame = FrameMetadata(
        frame_id="L23_V005_E001",
        video_id="L23_V005",
        shot_id="L23_V005_S000",
        timestamp_ms=1_000,
        fps=25.0,
        frame_idx=25,
        source="extracted",
        frame_path=Path("frame.jpg"),
    )
    result = OCRResult(
        frame_id=frame.frame_id,
        n=0,
        text="example",
        x_min=0.1,
        x_max=0.5,
        y_min=0.2,
        y_max=0.6,
        language="vi",
    )
    database = FakeDatabase([frame])
    service = FakeOCRService([result])

    results = run_ocr(  # type: ignore[arg-type]
        "L23_V005",
        database,
        service,
    )

    assert results == [result]
    assert service.received_frames == [frame]
    assert database.saved == [
        {
            "frame_id": frame.frame_id,
            "n": 0,
            "text": "example",
            "x_min": 0.1,
            "x_max": 0.5,
            "y_min": 0.2,
            "y_max": 0.6,
            "language": "vi",
        }
    ]
