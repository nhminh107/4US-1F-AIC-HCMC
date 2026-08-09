"""Fixture dùng chung cho toàn bộ test của Module OCR (module_ocr.md mục 12.1).

Nguyên tắc (mục 12.1): phần lớn test KHÔNG phụ thuộc GPU/model thật — dùng
``FakeOCREngine`` (implement đúng ``OCREngine`` Protocol) để test
``OCRExtractor`` độc lập với việc checkpoint MonkeyOCRv2 có sẵn hay không.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from BackEnd.app.contracts.pipeline import FrameMetadata
from BackEnd.app.ocr.engines.base import OCREngine, RawTextRegion

FIXTURES_DIR = Path(__file__).parent / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
EXPECTED_DIR = FIXTURES_DIR / "expected"


def pytest_configure(config: pytest.Config) -> None:
    """Đăng ký marker ``gpu`` (module_ocr.md mục 12.1, 12.9) để chạy ``pytest``
    không in cảnh báo "unknown marker" khi lọc bằng ``-m "not gpu"``/``-m gpu``.
    """

    config.addinivalue_line(
        "markers", "gpu: test cần checkpoint MonkeyOCRv2 thật + GPU, skip mặc định"
    )


class FakeOCREngine:
    """Engine giả lập tuân theo ``OCREngine`` Protocol, dùng cho mọi test không cần model thật.

    Có thể cấu hình để trả về 1 kết quả cố định (``regions``) mỗi lần gọi
    ``run()``, hoặc raise 1 lỗi cụ thể (``error``) để test đường lỗi của
    ``OCRExtractor``/``extract_batch``. Tự đếm số lần được gọi
    (``call_count``) để test xác nhận engine thực sự được (hoặc không được)
    gọi trong 1 số tình huống.
    """

    def __init__(
        self,
        regions: list[RawTextRegion] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.regions = regions if regions is not None else []
        self.error = error
        self.call_count = 0
        self.last_image: np.ndarray | None = None

    def run(self, image: np.ndarray) -> list[RawTextRegion]:
        self.call_count += 1
        self.last_image = image
        if self.error is not None:
            raise self.error
        return list(self.regions)


@pytest.fixture()
def fake_engine_factory():
    """Factory tạo ``FakeOCREngine`` mới cho từng test case, tránh chia sẻ state giữa các test."""

    def _factory(
        regions: list[RawTextRegion] | None = None, *, error: Exception | None = None
    ) -> FakeOCREngine:
        return FakeOCREngine(regions=regions, error=error)

    return _factory


@pytest.fixture()
def clear_text_image_path() -> Path:
    """Ảnh mẫu (1): có chữ rõ, tương phản cao."""

    return IMAGES_DIR / "clear_text.jpg"


@pytest.fixture()
def no_text_image_path() -> Path:
    """Ảnh mẫu (2): không có chữ."""

    return IMAGES_DIR / "no_text.jpg"


@pytest.fixture()
def blurry_small_text_image_path() -> Path:
    """Ảnh mẫu (3): chữ mờ/nhỏ."""

    return IMAGES_DIR / "blurry_small_text.jpg"


@pytest.fixture()
def ticker_banner_image_path() -> Path:
    """Ảnh mẫu (4): có banner/ticker chạy chữ ở cạnh dưới khung hình."""

    return IMAGES_DIR / "ticker_banner.jpg"


@pytest.fixture()
def make_frame_metadata(tmp_path: Path):
    """Factory tạo nhanh 1 ``FrameMetadata`` hợp lệ, cho phép override từng field.

    Mặc định ``frame_role="keyframe"`` và ``frame_path`` trỏ tới
    ``clear_text_image_path`` — đúng điều kiện tối thiểu để ``OCRExtractor``
    chấp nhận xử lý (module_ocr.md mục 2, mục 3).
    """

    def _factory(**overrides) -> FrameMetadata:
        defaults = dict(
            frame_id="F000001",
            video_id="L21_V001",
            shot_id="S000001",
            timestamp_ms=1000,
            fps=25.0,
            frame_idx=25,
            frame_role="keyframe",
            source="official",
            frame_path=IMAGES_DIR / "clear_text.jpg",
        )
        defaults.update(overrides)
        return FrameMetadata(**defaults)

    return _factory
