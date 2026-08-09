"""pHash dedup theo vùng tĩnh (module_ocr.md mục 7) — có sửa lỗi đã biết ở bản thiết kế cũ.

Ý tưởng giữ nguyên từ bản thiết kế cũ: nhiều keyframe liên tiếp có logo đài/
watermark/khung giờ ở VỊ TRÍ CỐ ĐỊNH → dùng perceptual hash (Hamming distance)
trên đúng vùng đó để nhận ra "vùng này chưa đổi so với lần trước", từ đó tái
sử dụng lại text đã nhận dạng được thay vì tin tưởng mù quáng kết quả mới mỗi
lần — hữu ích khi engine đọc sai lệch nhẹ (nhiễu nén ảnh) cùng 1 logo giữa các
frame khác nhau.

LỖI ĐÃ BIẾT cần fix (module_ocr.md mục 7): banner tin tức chạy chữ (news
ticker) có layout cố định nhưng NỘI DUNG đổi liên tục — nếu áp dụng pHash lên
cả vùng đó sẽ bị false positive (coi 2 nội dung khác nhau là trùng), làm mất
chữ mới. Cách fix ở đây: pHash CHỈ áp dụng cho các vùng được khai báo tường
minh trong ``config.DEFAULT_STATIC_REGIONS`` (hoặc override qua ``OCRConfig``)
— vùng ticker/banner đơn giản là KHÔNG được khai báo ở đó, nên không bao giờ
đi qua logic dedup trong file này, luôn giữ nguyên kết quả engine mới nhất.

LƯU Ý VỀ THỨ TỰ GỌI so với mục 5 của module_ocr.md: mô tả gốc liệt kê bước
dedup TRƯỚC bước gọi ``engine.run()``. Vì ``MonkeyOCREngine`` là model
end-to-end xử lý nguyên tấm ảnh trong 1 lần forward (không còn 2 bước
detect→crop→recognize tách rời như PaddleOCR+VietOCR trước đây), không thể
"bỏ qua" việc recognize riêng 1 vùng nhỏ trước khi biết toàn bộ ảnh có gì.
``OCRExtractor`` (xem ``ocr_extractor.py``) vì vậy áp dụng dedup NGAY SAU khi
có kết quả thô từ engine: nếu vùng tĩnh khớp cache, THAY THẾ text mới bằng
text đã cache (ổn định hơn, tránh sai lệch đọc logo/watermark giữa các
frame); nếu không khớp, giữ nguyên + cập nhật cache. Đây vẫn là 1 lớp tối ưu/
làm ổn định kết quả, không phải điều kiện bắt buộc để pipeline chạy đúng
(mục 8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from BackEnd.app.ocr import config
from BackEnd.app.ocr.config import StaticRegion
from BackEnd.app.ocr.engines.base import RawTextRegion
from BackEnd.app.ocr.preprocess import crop_static_region

logger = logging.getLogger(__name__)


def perceptual_hash(image: np.ndarray, hash_size: int = 8) -> int:
    """Tính average-hash (aHash) cho 1 ảnh: thu nhỏ về grayscale ``hash_size x hash_size``,
    so từng điểm ảnh với giá trị trung bình để ra 1 chuỗi bit, gộp thành số nguyên.

    2 ảnh giống nhau (hoặc gần giống — sai khác nhỏ do nén/anti-aliasing) sẽ
    cho ra hash giống nhau hoặc lệch rất ít bit.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    average = float(resized.mean())

    bits = 0
    for value in resized.flatten():
        bits = (bits << 1) | int(value > average)
    return bits


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Số bit khác nhau giữa 2 hash (XOR rồi đếm số bit 1)."""

    return bin(hash_a ^ hash_b).count("1")


def is_duplicate_hash(hash_a: int, hash_b: int, threshold: int = config.PHASH_HAMMING_THRESHOLD) -> bool:
    """2 hash được coi là "cùng 1 nội dung" nếu Hamming distance không vượt ngưỡng."""

    return hamming_distance(hash_a, hash_b) <= threshold


@dataclass(slots=True)
class _CacheEntry:
    """1 lần quan sát trước đó của 1 static region: hash tại thời điểm đó + text đã nhận dạng."""

    phash: int
    regions: list[RawTextRegion]


@dataclass(slots=True)
class StaticRegionDedupCache:
    """Cache (hash -> regions) cho từng vùng tĩnh đã khai báo, dùng chung xuyên suốt 1 lần chạy.

    Duyệt tuyến tính để so khớp Hamming distance khi tra cứu — chấp nhận được
    vì số lượng static_region khai báo trong config nhỏ (vài logo/watermark),
    dù số keyframe xử lý có thể rất lớn.
    """

    threshold: int = config.PHASH_HAMMING_THRESHOLD
    _entries_by_region: dict[str, list[_CacheEntry]] = field(default_factory=dict)

    def lookup(self, region_name: str, image_hash: int) -> list[RawTextRegion] | None:
        """Trả về danh sách ``RawTextRegion`` đã cache nếu tìm thấy hash gần trùng, ngược lại ``None``."""

        for entry in self._entries_by_region.get(region_name, []):
            if is_duplicate_hash(entry.phash, image_hash, self.threshold):
                return entry.regions
        return None

    def update(self, region_name: str, image_hash: int, regions: list[RawTextRegion]) -> None:
        """Ghi nhận 1 quan sát mới cho vùng ``region_name``."""

        self._entries_by_region.setdefault(region_name, []).append(
            _CacheEntry(phash=image_hash, regions=regions)
        )

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._entries_by_region.values())


def apply_static_region_dedup(
    image: np.ndarray,
    raw_regions: list[RawTextRegion],
    static_regions: tuple[StaticRegion, ...],
    cache: StaticRegionDedupCache,
    *,
    overlap_ratio: float = config.STATIC_REGION_OVERLAP_RATIO,
) -> list[RawTextRegion]:
    """Đối chiếu kết quả thô của engine với cache pHash theo từng static_region đã khai báo.

    Với mỗi ``StaticRegion`` trong config: tính pHash của đúng vùng ảnh đó,
    tìm các ``RawTextRegion`` do engine trả về nằm chồng lấn đủ nhiều
    (``overlap_ratio``) lên vùng này. Nếu hash khớp 1 quan sát cũ trong cache
    → thay các vùng khớp bằng kết quả đã cache (ổn định hơn); nếu không khớp
    → giữ nguyên kết quả engine vừa trả về và ghi thêm 1 quan sát mới vào
    cache. Các ``RawTextRegion`` KHÔNG nằm trong bất kỳ static_region nào
    (gồm cả toàn bộ vùng ticker/banner, vì không được khai báo) luôn được giữ
    nguyên như engine trả về.
    """

    if not static_regions or not raw_regions:
        return raw_regions

    image_height, image_width = image.shape[:2]

    matched_indices: set[int] = set()
    substituted_regions: list[RawTextRegion] = []

    for region in static_regions:
        region_bbox_px = _static_region_to_pixel_bbox(region, image_width, image_height)

        crop = crop_static_region(image, region)
        if crop.size == 0:
            # Toạ độ khai báo không cắt ra được vùng ảnh hợp lệ (vd config
            # dùng chung cho video có tỉ lệ khung hình khác biệt lớn) - bỏ
            # qua vùng này, không làm crash cả frame.
            logger.warning(
                "dedup: static_region '%s' cắt ra ảnh rỗng trên frame kích thước %sx%s, bỏ qua",
                region.name,
                image_width,
                image_height,
            )
            continue

        region_matches = [
            i
            for i, raw_region in enumerate(raw_regions)
            if i not in matched_indices
            and _overlap_ratio(raw_region.bbox, region_bbox_px) >= overlap_ratio
        ]
        if not region_matches:
            continue

        matched_indices.update(region_matches)
        current_regions = [raw_regions[i] for i in region_matches]

        region_hash = perceptual_hash(crop)
        cached_regions = cache.lookup(region.name, region_hash)
        if cached_regions is not None:
            logger.debug("dedup: tái sử dụng text đã cache cho static_region '%s'", region.name)
            substituted_regions.extend(cached_regions)
        else:
            cache.update(region.name, region_hash, current_regions)
            substituted_regions.extend(current_regions)

    unmatched_regions = [
        raw_region for i, raw_region in enumerate(raw_regions) if i not in matched_indices
    ]
    return unmatched_regions + substituted_regions


def _static_region_to_pixel_bbox(
    region: StaticRegion, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Quy đổi 1 ``StaticRegion`` (normalized) sang bbox pixel theo đúng kích thước ảnh hiện tại."""

    return (
        region.x_min * image_width,
        region.y_min * image_height,
        region.x_max * image_width,
        region.y_max * image_height,
    )


def _overlap_ratio(
    bbox: tuple[float, float, float, float], other_bbox: tuple[float, float, float, float]
) -> float:
    """Tỉ lệ diện tích giao nhau của ``bbox`` so với chính diện tích ``bbox`` (không đối xứng).

    Cố ý KHÔNG dùng IoU đối xứng: mục tiêu là biết "vùng chữ này có nằm gọn
    trong static_region không", nên chuẩn hoá theo diện tích của ``bbox``
    (vùng chữ do engine trả về) chứ không phải theo hợp của 2 vùng.
    """

    ax_min, ay_min, ax_max, ay_max = bbox
    bx_min, by_min, bx_max, by_max = other_bbox

    inter_width = max(0.0, min(ax_max, bx_max) - max(ax_min, bx_min))
    inter_height = max(0.0, min(ay_max, by_max) - max(ay_min, by_min))
    intersection_area = inter_width * inter_height

    bbox_area = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    if bbox_area <= 0:
        return 0.0
    return intersection_area / bbox_area
