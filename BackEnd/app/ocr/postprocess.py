"""Hậu xử lý kết quả thô của engine, trước khi đóng gói thành ``OCRResult`` (module_ocr.md mục 5, 12.7).

3 việc độc lập, áp dụng theo đúng thứ tự trong ``postprocess()``:
1. Chuẩn hoá từng chuỗi text (xoá ký tự điều khiển, gộp khoảng trắng thừa).
2. Gộp các vùng nằm trên cùng 1 "hàng" chữ (trường hợp engine trả tách nhỏ
   nhiều mảnh của cùng 1 dòng/khối logic) thành 1 vùng duy nhất, không làm
   mất thứ tự đọc.
3. Lọc bỏ vùng có confidence thấp hơn ngưỡng cấu hình.

Toàn bộ hàm ở đây làm việc trên ``RawTextRegion`` (bbox pixel) — việc quy đổi
sang normalized ``[0, 1]`` và gán ``n`` để thành ``OCRResult`` thuộc trách
nhiệm của ``OCRExtractor`` (qua ``bbox_utils.py``), tách biệt rõ 2 việc.
"""

from __future__ import annotations

import re

from BackEnd.app.ocr import config
from BackEnd.app.ocr.engines.base import RawTextRegion

# Ký tự điều khiển (control character) đôi khi lẫn trong output thô của model
# sinh text (vd \x00, \x0b) — không có giá trị hiển thị, cần loại trước khi
# gộp khoảng trắng.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Xoá ký tự điều khiển, gộp nhiều khoảng trắng liên tiếp thành 1, xoá khoảng trắng đầu/cuối."""

    without_control_chars = _CONTROL_CHARS_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", without_control_chars).strip()


def sort_regions_reading_order(
    regions: list[RawTextRegion], *, line_y_tolerance: int = config.LINE_Y_TOLERANCE_PX
) -> list[RawTextRegion]:
    """Sắp xếp các vùng chữ theo thứ tự đọc tự nhiên: trên -> dưới, trong cùng hàng thì trái -> phải.

    Gộp theo "hàng" dựa trên ``y_min`` (cho phép sai lệch nhỏ trong
    ``line_y_tolerance`` pixel, để không sắp sai thứ tự do chênh lệch nhỏ
    giữa các bbox trên cùng 1 dòng — vd ký tự có dấu cao hơn ký tự thường).
    """

    def sort_key(region: RawTextRegion) -> tuple[int, float]:
        x_min, y_min = region.bbox[0], region.bbox[1]
        return (round(y_min / line_y_tolerance), x_min)

    return sorted(regions, key=sort_key)


def merge_same_line_regions(
    regions: list[RawTextRegion], *, line_y_tolerance: int = config.LINE_Y_TOLERANCE_PX
) -> list[RawTextRegion]:
    """Gộp các ``RawTextRegion`` nằm trên cùng 1 hàng thành 1 vùng duy nhất.

    Dùng cho trường hợp engine trả về nhiều mảnh nhỏ tách rời của cùng 1 dòng/
    khối chữ logic (vd do layout detector chia quá nhỏ) — gộp lại theo đúng
    thứ tự đọc (trái -> phải trong hàng) thay vì giữ nguyên nhiều ``OCRResult``
    rời rạc gây nhiễu kết quả tìm kiếm.

    Bbox của vùng gộp là hợp (union) bbox của các vùng con. Confidence lấy
    giá trị THẤP NHẤT trong nhóm (1 khối chỉ đáng tin bằng thành phần yếu
    nhất của nó). Language chỉ giữ lại nếu mọi vùng con đồng nhất — khác
    nhau thì để ``None`` thay vì đoán bừa (cùng nguyên tắc mục 9).
    """

    if not regions:
        return []

    ordered = sort_regions_reading_order(regions, line_y_tolerance=line_y_tolerance)

    merged: list[RawTextRegion] = []
    current_group = [ordered[0]]
    current_row = round(ordered[0].bbox[1] / line_y_tolerance)

    for region in ordered[1:]:
        row = round(region.bbox[1] / line_y_tolerance)
        if row == current_row:
            current_group.append(region)
            continue
        merged.append(_merge_group(current_group))
        current_group = [region]
        current_row = row

    merged.append(_merge_group(current_group))
    return merged


def filter_by_confidence(
    regions: list[RawTextRegion], threshold: float = config.CONFIDENCE_THRESHOLD
) -> list[RawTextRegion]:
    """Loại bỏ vùng có ``confidence`` thấp hơn ngưỡng.

    Vùng không có confidence (``None`` — engine không trả kèm điểm tin cậy)
    được GIỮ LẠI thay vì loại: không đủ căn cứ để coi là "thấp", loại bừa sẽ
    làm mất text hợp lệ chỉ vì thiếu metadata.
    """

    return [region for region in regions if region.confidence is None or region.confidence >= threshold]


def postprocess(
    regions: list[RawTextRegion],
    *,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    line_y_tolerance: int = config.LINE_Y_TOLERANCE_PX,
) -> list[RawTextRegion]:
    """Áp dụng đầy đủ chuỗi hậu xử lý: chuẩn hoá text -> gộp dòng -> lọc confidence -> sắp lại thứ tự đọc."""

    normalized = [
        RawTextRegion(
            text=normalize_text(region.text),
            bbox=region.bbox,
            confidence=region.confidence,
            language=region.language,
        )
        for region in regions
    ]
    non_empty = [region for region in normalized if region.text]

    merged = merge_same_line_regions(non_empty, line_y_tolerance=line_y_tolerance)
    filtered = filter_by_confidence(merged, confidence_threshold)
    return sort_regions_reading_order(filtered, line_y_tolerance=line_y_tolerance)


def _merge_group(group: list[RawTextRegion]) -> RawTextRegion:
    """Gộp 1 nhóm ``RawTextRegion`` (đã cùng hàng, đã sắp trái -> phải) thành 1 region."""

    text = " ".join(region.text for region in group if region.text)

    x_min = min(region.bbox[0] for region in group)
    y_min = min(region.bbox[1] for region in group)
    x_max = max(region.bbox[2] for region in group)
    y_max = max(region.bbox[3] for region in group)

    confidences = [region.confidence for region in group if region.confidence is not None]
    confidence = min(confidences) if confidences else None

    languages = {region.language for region in group if region.language}
    language = next(iter(languages)) if len(languages) == 1 else None

    return RawTextRegion(text=text, bbox=(x_min, y_min, x_max, y_max), confidence=confidence, language=language)
