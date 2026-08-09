"""Chuyển đổi + kiểm tra bounding box cho Module OCR (module_ocr.md mục 3).

``OCREngine`` trả về bbox theo toạ độ pixel trên ảnh đã tiền xử lý (resize).
Trước khi đóng gói thành ``OCRResult`` (data contract ``pipeline.py``),
``OCRExtractor`` phải quy đổi sang toạ độ normalized ``[0, 1]`` và đảm bảo
đúng ràng buộc CHECK của bảng ``OCR`` trong ``postgre_script.sql``:

    x_min, x_max, y_min, y_max in [0, 1]
    x_min < x_max
    y_min < y_max

Module này tách riêng phần "toán" thuần tuý (không phụ thuộc engine/ảnh cụ
thể) để dễ test độc lập (``tests/ocr/test_bbox_utils.py``).
"""

from __future__ import annotations


class InvalidBBoxError(ValueError):
    """Bbox không thể chuyển thành 1 vùng hợp lệ theo ràng buộc bảng ``OCR``.

    Được raise thay vì âm thầm bỏ qua, để nơi gọi (``OCRExtractor.extract``)
    tự quyết định: loại riêng vùng lỗi này (log rõ frame_id) và giữ lại các
    vùng còn lại hợp lệ — không được để 1 bbox hỏng làm crash toàn bộ frame.
    """


def clip_pixel_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Giới hạn 1 bbox toạ độ pixel về trong khung ảnh ``[0, width] x [0, height]``.

    Engine có thể trả về toạ độ hơi vượt biên ảnh (sai số làm tròn, model dự
    đoán lố mép) — clip về đúng khung ảnh thay vì raise lỗi ngay, vì đây là
    sai lệch nhỏ có thể chấp nhận được, khác với trường hợp bbox suy biến
    (min >= max) sau khi clip, lúc đó mới coi là lỗi thật sự (xem
    ``pixel_to_normalized``).
    """

    clipped_x_min = min(max(x_min, 0.0), float(image_width))
    clipped_x_max = min(max(x_max, 0.0), float(image_width))
    clipped_y_min = min(max(y_min, 0.0), float(image_height))
    clipped_y_max = min(max(y_max, 0.0), float(image_height))
    return clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max


def pixel_to_normalized(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Quy đổi 1 bbox pixel -> normalized ``[0, 1]`` theo đúng công thức ``x/width``, ``y/height``.

    Trả về ``(x_min, x_max, y_min, y_max)`` normalized. Raise
    ``InvalidBBoxError`` nếu bbox nằm hoàn toàn ngoài khung ảnh (sau khi clip
    bị suy biến thành ``x_min >= x_max`` hoặc ``y_min >= y_max``) — vùng như
    vậy không có phần diện tích nào hợp lệ để giữ lại.
    """

    if image_width <= 0 or image_height <= 0:
        raise InvalidBBoxError(
            f"Kích thước ảnh không hợp lệ để quy đổi bbox: "
            f"width={image_width}, height={image_height}"
        )

    clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max = clip_pixel_bbox(
        x_min, y_min, x_max, y_max, image_width=image_width, image_height=image_height
    )

    norm_x_min = clipped_x_min / image_width
    norm_x_max = clipped_x_max / image_width
    norm_y_min = clipped_y_min / image_height
    norm_y_max = clipped_y_max / image_height

    # Sau khi clip, ép lại vào đúng [0, 1] để tránh sai số dấu phẩy động (vd
    # 1.0000000002) làm fail CHECK constraint `between 0 and 1` khi ghi DB.
    norm_x_min, norm_x_max = _clamp01(norm_x_min), _clamp01(norm_x_max)
    norm_y_min, norm_y_max = _clamp01(norm_y_min), _clamp01(norm_y_max)

    validate_normalized_bbox(norm_x_min, norm_x_max, norm_y_min, norm_y_max)
    return norm_x_min, norm_x_max, norm_y_min, norm_y_max


def validate_normalized_bbox(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Kiểm tra 1 bbox normalized đúng ràng buộc bảng ``OCR`` (postgre_script.sql).

    Raise ``InvalidBBoxError`` kèm giá trị cụ thể nếu vi phạm, để log/debug dễ
    truy vết thay vì chỉ báo "bbox invalid" chung chung.
    """

    for name, value in (("x_min", x_min), ("x_max", x_max), ("y_min", y_min), ("y_max", y_max)):
        if not 0.0 <= value <= 1.0:
            raise InvalidBBoxError(f"{name}={value} nằm ngoài khoảng [0, 1]")

    if x_min >= x_max:
        raise InvalidBBoxError(f"x_min ({x_min}) phải nhỏ hơn x_max ({x_max})")
    if y_min >= y_max:
        raise InvalidBBoxError(f"y_min ({y_min}) phải nhỏ hơn y_max ({y_max})")


def _clamp01(value: float) -> float:
    """Ép 1 giá trị float về đúng khoảng đóng [0, 1]."""

    return min(max(value, 0.0), 1.0)
