"""Sample N frame đại diện trong khoảng thời gian của 1 Clip.

Phục vụ Clip Caption (``Markdown_Doc/module_caption.md`` mục 3.2, hướng
"Multi-frame sampling" đã chốt): thay vì chạy 1 model Video-VLM riêng, ta
chọn ra vài frame tĩnh trải đều theo thời gian trong clip rồi feed cả loạt
vào 1 lần gọi VLM đa ảnh.

Tách thành hàm thuần (không import model/VLM) để test được độc lập, không
cần chạy inference thật (module_caption.md mục 5).
"""

from __future__ import annotations

import numpy as np

from BackEnd.app.caption import config
from BackEnd.app.contracts.pipeline import ClipWindowMetadata, FrameMetadata


def sample_frames_in_range(
    clip: ClipWindowMetadata,
    frames: list[FrameMetadata],
    num_samples: int = config.DEFAULT_CLIP_SAMPLE_COUNT,
) -> list[FrameMetadata]:
    """Chọn tối đa ``num_samples`` frame nằm trong khoảng nửa-mở ``[clip.start_ms, clip.end_ms)``.

    Khoảng nửa-mở được dùng nhất quán với quy ước thời gian đã chốt cho Shot
    (xem ``shot_extractor_pipeline.md`` mục 6): 2 clip liền kề không tranh
    nhau cùng 1 frame nằm đúng trên ranh giới ``end_ms``.

    Chỉ chọn TRONG SỐ frame được truyền vào (``frames``), không tự extract
    thêm — việc extract thêm khi khoảng clip quá thưa frame là trách nhiệm
    của orchestrator (``module_caption.md`` mục 3.2), không phải của hàm
    thuần này.

    Nếu số frame khớp trong khoảng đã <= ``num_samples``, trả về toàn bộ
    (theo đúng thứ tự thời gian) — không nhân bản cho đủ số lượng, vì nhân
    bản sẽ khiến VLM nhận nhiều ảnh giống hệt nhau và hiểu lầm là clip có
    nhiều khung hình quan sát hơn thực tế.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples phải > 0, nhận được {num_samples}")

    candidates = sorted(
        (frame for frame in frames if clip.start_ms <= frame.timestamp_ms < clip.end_ms),
        key=lambda frame: frame.timestamp_ms,
    )

    if len(candidates) <= num_samples:
        return candidates

    # Chọn chỉ số đều nhau trên danh sách đã sort theo thời gian (uniform
    # sampling theo thời gian). set() để loại chỉ số trùng khi làm tròn dồn
    # vào cùng 1 vị trí (xảy ra khi num_samples gần bằng len(candidates)).
    raw_indices = np.linspace(0, len(candidates) - 1, num_samples).round().astype(int)
    unique_sorted_indices = sorted(set(raw_indices.tolist()))

    return [candidates[i] for i in unique_sorted_indices]
