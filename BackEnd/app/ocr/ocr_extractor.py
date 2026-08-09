"""``OCRExtractor`` — entry point duy nhất của Module OCR (module_ocr.md mục 3, 5).

Input: 1 ``FrameMetadata`` (``app/contracts/pipeline.py``) — bắt buộc
``frame_role == "keyframe"`` và có ``frame_path`` trỏ tới ảnh đọc được.
Output: ``list[OCRResult]``, rỗng nếu frame không có chữ.

Theo Q1 (``THỐNG_NHẤT_CHUNG``, nhắc lại ở module_ocr.md mục 1, 10):
``OCRExtractor`` chỉ làm input -> output. KHÔNG tự ghi PostgreSQL, KHÔNG tự
index Elasticsearch, KHÔNG import bất kỳ gì từ ``app/database/`` — việc đó do
``app/pipeline/`` đảm nhiệm sau khi gọi module này.

Pipeline nội bộ của ``extract()`` (module_ocr.md mục 5):
1. Đọc ảnh từ ``frame.frame_path``.
2. Tiền xử lý (resize + tăng tương phản — ``preprocess.py``).
3. Gọi ``engine.run()`` (``MonkeyOCREngine`` — detect + recognize 1 lần).
4. Đối chiếu pHash dedup theo vùng tĩnh (``dedup.py``) — ổn định hoá text của
   logo/watermark cố định, xem lý do đặt SAU bước gọi engine (khác thứ tự mô
   tả gốc mục 5) trong docstring của ``dedup.apply_static_region_dedup``.
5. Hậu xử lý text (``postprocess.py``): chuẩn hoá chuỗi, gộp dòng, lọc theo
   confidence.
6. Quy đổi bbox pixel -> normalized ``[0, 1]`` (``bbox_utils.py``), gán ``n``
   tăng dần từ 0, build ``OCRResult``, bỏ qua (kèm log) riêng vùng nào có bbox
   không hợp lệ thay vì làm hỏng cả frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult
from BackEnd.app.ocr import config as ocr_config
from BackEnd.app.ocr.bbox_utils import InvalidBBoxError, pixel_to_normalized
from BackEnd.app.ocr.config import OCRConfig
from BackEnd.app.ocr.dedup import StaticRegionDedupCache, apply_static_region_dedup
from BackEnd.app.ocr.engines.base import OCREngine, RawTextRegion
from BackEnd.app.ocr.postprocess import postprocess
from BackEnd.app.ocr.preprocess import load_image
from BackEnd.app.ocr.preprocess import preprocess as preprocess_image

logger = logging.getLogger(__name__)

# frame_role duy nhất module này được phép xử lý (module_ocr.md mục 2):
# tracking_sample không có frame_path (xem CHECK constraint bảng Frame trong
# postgre_script.sql), nên không có ảnh để OCR.
_REQUIRED_FRAME_ROLE = "keyframe"


class OCRExtractor:
    """Trích xuất text xuất hiện trong keyframe, trả về theo đúng contract ``OCRResult``."""

    def __init__(self, engine: OCREngine, config: OCRConfig | None = None) -> None:
        """Khởi tạo extractor với 1 engine cụ thể (hiện tại: ``MonkeyOCREngine``).

        Args:
            engine: Engine OCR tuân theo ``OCREngine`` Protocol
                (``engines/base.py``). Cho phép truyền vào từ ngoài chủ yếu để
                test (mock engine không cần model/GPU thật) và để giữ đúng
                ranh giới kiến trúc đã chốt ở module_ocr.md mục 4.3/5.
            config: Bộ tham số tinh chỉnh (ngưỡng confidence, static_regions,
                resize...). Mặc định dùng ``ocr_config.DEFAULT_CONFIG``.
        """

        self._engine = engine
        self._config = config if config is not None else ocr_config.DEFAULT_CONFIG

        # Cache dedup dùng CHUNG xuyên suốt vòng đời của instance (không reset
        # theo từng lần gọi extract()) - logo/watermark đài lặp lại xuyên
        # suốt cả video, không chỉ trong 1 lần gọi.
        self._dedup_cache = StaticRegionDedupCache(threshold=self._config.phash_hamming_threshold)

    def extract(self, frame: FrameMetadata) -> list[OCRResult]:
        """Trích xuất text cho đúng 1 keyframe.

        Raise ``ValueError`` nếu ``frame.frame_role`` không phải
        ``"keyframe"`` hoặc thiếu ``frame_path``; raise ``FileNotFoundError``
        nếu ``frame_path`` không trỏ tới file đọc được — đây là lỗi cấu hình
        gọi module SAI cách, KHÔNG được âm thầm trả về ``[]`` (khác với
        trường hợp "frame có ảnh hợp lệ nhưng không có chữ", vốn là kết quả
        hợp lệ trả về ``[]``, xem module_ocr.md mục 3).
        """

        _validate_frame_role(frame)
        image_path = _validate_frame_path(frame)

        raw_image = load_image(image_path)
        processed_image = preprocess_image(raw_image, max_side=self._config.resize_max_side)

        raw_regions = self._engine.run(processed_image)
        if not raw_regions:
            return []

        deduped_regions = apply_static_region_dedup(
            processed_image,
            raw_regions,
            self._config.static_regions,
            self._dedup_cache,
            overlap_ratio=self._config.static_region_overlap_ratio,
        )

        final_regions = postprocess(
            deduped_regions,
            confidence_threshold=self._config.confidence_threshold,
            line_y_tolerance=self._config.line_y_tolerance_px,
        )

        return _build_ocr_results(final_regions, frame, processed_image)

    def extract_batch(self, frames: list[FrameMetadata]) -> dict[str, list[OCRResult]]:
        """Chạy ``extract()`` cho nhiều frame — bắt buộc dùng cho pha offline indexing
        (module_ocr.md mục 5, mục 8: số lượng keyframe rất lớn, không gọi ``extract()``
        tuần tự thủ công từng frame ở tầng orchestrator).

        Cô lập lỗi theo từng frame (module_ocr.md mục 12.2): 1 frame lỗi (thiếu
        ảnh, bbox hỏng, engine crash...) chỉ làm frame đó nhận kết quả rỗng kèm
        log lỗi đầy đủ ``frame_id`` + lý do — KHÔNG làm crash toàn bộ batch. Vì
        đây chính là ranh giới cô lập lỗi có chủ đích của hàm này (khác
        ``extract()``, nơi lỗi cấu hình phải raise ngay), bắt ``Exception``
        rộng ở đây là lựa chọn thiết kế tường minh, không phải "nuốt lỗi âm
        thầm" — mọi lỗi đều được log kèm danh tính frame trước khi tiếp tục.
        """

        results: dict[str, list[OCRResult]] = {}
        for frame in frames:
            try:
                results[frame.frame_id] = self.extract(frame)
            except Exception as exc:
                logger.error(
                    "extract_batch: frame %s lỗi khi OCR (%s), gán kết quả rỗng và tiếp tục batch",
                    frame.frame_id,
                    exc,
                )
                results[frame.frame_id] = []
        return results


def _validate_frame_role(frame: FrameMetadata) -> None:
    """Chỉ chấp nhận frame_role == 'keyframe' (module_ocr.md mục 2)."""

    if frame.frame_role != _REQUIRED_FRAME_ROLE:
        raise ValueError(
            f"OCRExtractor chỉ xử lý frame_role='{_REQUIRED_FRAME_ROLE}', nhận được "
            f"frame_id={frame.frame_id!r} với frame_role={frame.frame_role!r}. "
            "tracking_sample không có frame_path để đọc ảnh (postgre_script.sql, bảng Frame)."
        )


def _validate_frame_path(frame: FrameMetadata) -> Path:
    """Kiểm tra ``frame.frame_path`` tồn tại và trỏ tới 1 file thật, trả về ``Path`` đã kiểm tra."""

    if frame.frame_path is None:
        raise ValueError(f"Frame {frame.frame_id!r} không có frame_path, không thể OCR.")

    path = Path(frame.frame_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frame {frame.frame_id!r}: không tìm thấy file ảnh tại {path}")
    return path


def _build_ocr_results(
    regions: list[RawTextRegion], frame: FrameMetadata, image: np.ndarray
) -> list[OCRResult]:
    """Quy đổi bbox pixel -> normalized, gán ``n`` tăng dần, đóng gói thành ``OCRResult``.

    Vùng nào có bbox không hợp lệ sau khi quy đổi (``InvalidBBoxError``) bị bỏ
    qua kèm log rõ ``frame_id`` — không làm hỏng các vùng còn lại của cùng
    frame (module_ocr.md mục 3: ràng buộc bbox phải tự kiểm tra trước khi trả
    về, khớp CHECK constraint bảng ``OCR``).
    """

    image_height, image_width = image.shape[:2]

    results: list[OCRResult] = []
    next_n = 0
    for region in regions:
        try:
            x_min, x_max, y_min, y_max = pixel_to_normalized(
                *region.bbox, image_width=image_width, image_height=image_height
            )
        except InvalidBBoxError as exc:
            logger.warning(
                "extract: frame %s bỏ qua 1 vùng chữ do bbox không hợp lệ sau khi quy đổi (%s), "
                "text=%r",
                frame.frame_id,
                exc,
                region.text,
            )
            continue

        results.append(
            OCRResult(
                frame_id=frame.frame_id,
                n=next_n,
                text=region.text,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                language=region.language,
            )
        )
        next_n += 1

    return results
