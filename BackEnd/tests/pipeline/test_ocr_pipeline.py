from __future__ import annotations

from pathlib import Path

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.pipeline.ocr import run_ocr


class FakeDatabase:
    def __init__(self, frames: list[FrameMetadata]) -> None:
        self.frames = frames
        self.saved: list[OCRResult] = []

    def get_frame_record_by_video_id(self, video_id: str) -> list[FrameMetadata]:
        assert video_id == "L23_V005"
        return self.frames

    def add_ocr_records(self, results: list[OCRResult]) -> None:
        self.saved.extend(results)


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
    assert database.saved == [result]


def test_run_ocr_bounds_frame_memory_with_chunks() -> None:
    frames = [
        FrameMetadata(
            frame_id=f"L23_V005_E{index:03d}",
            video_id="L23_V005",
            shot_id="L23_V005_S000",
            timestamp_ms=index * 1_000,
            fps=25.0,
            frame_idx=index * 25,
            source="extracted",
            frame_path=Path(f"frame-{index}.jpg"),
        )
        for index in range(1, 4)
    ]

    class ChunkedService:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def process_batch(self, batch: list[FrameMetadata]) -> list[OCRResult]:
            self.batches.append([frame.frame_id for frame in batch])
            return [
                OCRResult(
                    frame_id=frame.frame_id,
                    n=0,
                    text="text",
                    x_min=0.0,
                    x_max=1.0,
                    y_min=0.0,
                    y_max=1.0,
                    language="vi",
                )
                for frame in batch
            ]

    database = FakeDatabase(frames)
    service = ChunkedService()

    results = run_ocr(  # type: ignore[arg-type]
        "L23_V005",
        database,
        service,  # type: ignore[arg-type]
        frame_chunk_size=2,
    )

    assert service.batches == [
        ["L23_V005_E001", "L23_V005_E002"],
        ["L23_V005_E003"],
    ]
    assert len(results) == 3
    assert len(database.saved) == 3


def test_run_ocr_filters_frames_by_source() -> None:
    official = FrameMetadata(
        frame_id="L23_V005_001",
        video_id="L23_V005",
        shot_id="L23_V005_S000",
        timestamp_ms=1_000,
        fps=25.0,
        frame_idx=25,
        source="official",
        frame_path=Path("official.jpg"),
    )
    extracted = FrameMetadata(
        frame_id="L23_V005_E001",
        video_id="L23_V005",
        shot_id="L23_V005_S000",
        timestamp_ms=1_500,
        fps=25.0,
        frame_idx=38,
        source="extracted",
        frame_path=Path("extracted.jpg"),
    )
    database = FakeDatabase([official, extracted])
    service = FakeOCRService([])

    results = run_ocr(  # type: ignore[arg-type]
        "L23_V005",
        database,
        service,
        frame_source="official",
    )

    assert results == []
    assert service.received_frames == [official]
