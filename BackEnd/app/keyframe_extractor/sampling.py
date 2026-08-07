"""Logic thuần chọn candidate frame indices cho keyframe bổ sung (additional keyframe).

Module này chứa toán học và thuật toán chọn frame index thuần túy (dùng
Python/NumPy thuần), không phụ thuộc vào FFmpeg, OpenCV, hay filesystem. Nhờ
đó có thể unit-test cực kỳ nhanh và độc lập.
"""

from __future__ import annotations

from collections.abc import Sequence, Set

DEFAULT_TARGET_INTERVAL_MS = 2500  # Mốc thời gian mặc định: ~2.5s / 1 keyframe
DEFAULT_MIN_FRAME_GAP = 5          # Khoảng cách tối thiểu giữa 2 keyframe (tránh gần nhau)
DEFAULT_MAX_ADDITIONAL_PER_SHOT = 5 # Tối đa số keyframe bổ sung trích xuất cho 1 shot


def select_additional_keyframe_indices(
    start_frame_idx: int,
    end_frame_idx: int,
    start_ms: int,
    end_ms: int,
    fps: float,
    existing_frame_idxs: Sequence[int] | Set[int] | None = None,
    *,
    min_frame_gap: int = DEFAULT_MIN_FRAME_GAP,
    target_interval_ms: int = DEFAULT_TARGET_INTERVAL_MS,
    max_additional_per_shot: int = DEFAULT_MAX_ADDITIONAL_PER_SHOT,
) -> list[int]:
    """Chọn các candidate `frame_idx` bổ sung cho một shot.

    Args:
        start_frame_idx: Frame index bắt đầu của shot (inclusive).
        end_frame_idx: Frame index kết thúc của shot (inclusive).
        start_ms: Thời điểm bắt đầu shot (ms).
        end_ms: Thời điểm kết thúc shot (ms).
        fps: Frame rate của video (> 0).
        existing_frame_idxs: Danh sách/tập hợp các `frame_idx` keyframe đã có
            gốc (official) của video.
        min_frame_gap: Khoảng cách frame tối thiểu không được nằm quá gần các
            keyframe đã có hoặc candidate đã chọn.
        target_interval_ms: Khoảng thời gian mục tiêu (ms) giữa các keyframe.
        max_additional_per_shot: Số lượng keyframe trích xuất thêm tối đa cho 1 shot.

    Returns:
        Danh sách các `frame_idx` bổ sung được chọn, sắp xếp tăng dần và không
        trùng lặp với keyframe official.
    """

    if start_frame_idx < 0:
        raise ValueError(f"start_frame_idx must be >= 0, got {start_frame_idx}")
    if end_frame_idx < start_frame_idx:
        raise ValueError(
            f"end_frame_idx ({end_frame_idx}) must be >= start_frame_idx ({start_frame_idx})"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    existing_set: set[int] = set(existing_frame_idxs) if existing_frame_idxs is not None else set()

    # Tìm các keyframe official đã nằm sẵn trong shot này
    shot_existing = {f for f in existing_set if start_frame_idx <= f <= end_frame_idx}

    duration_ms = max(1, end_ms - start_ms)

    # Tính số keyframe mục tiêu cho shot này
    if duration_ms < target_interval_ms:
        target_count = 1
    else:
        target_count = 1 + int(duration_ms // target_interval_ms)
        target_count = min(target_count, 1 + max_additional_per_shot)

    # Nếu số keyframe sẵn có đã đủ hoặc vượt target -> không cần lấy thêm
    if len(shot_existing) >= target_count:
        return []

    needed_count = target_count - len(shot_existing)
    total_frames = end_frame_idx - start_frame_idx + 1

    # Chia đều shot thành target_count phần và lấy điểm trung tâm mỗi phần
    candidates: list[int] = []
    forbidden_set = set(existing_set)

    for i in range(target_count):
        # Center index của phần thứ i
        center_rel = int((2 * i + 1) * total_frames / (2 * target_count))
        proposed = start_frame_idx + center_rel
        proposed = min(max(proposed, start_frame_idx), end_frame_idx)

        # Kiểm tra xung đột với forbidden_set (hoặc khoảng min_frame_gap)
        valid_idx = _find_nearest_valid_idx(
            proposed,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
            forbidden_set=forbidden_set,
            min_frame_gap=min_frame_gap,
        )

        if valid_idx is not None:
            candidates.append(valid_idx)
            # Thêm các frame xung quanh valid_idx vào forbidden_set để candidate sau không dính sát
            for f in range(valid_idx - min_frame_gap, valid_idx + min_frame_gap + 1):
                forbidden_set.add(f)

        if len(candidates) >= needed_count:
            break

    return sorted(candidates)


def _find_nearest_valid_idx(
    target: int,
    *,
    start_frame_idx: int,
    end_frame_idx: int,
    forbidden_set: set[int],
    min_frame_gap: int,
) -> int | None:
    """Tìm frame index hợp lệ gần `target` nhất trong khoảng `[start_frame_idx, end_frame_idx]`."""

    def is_valid(idx: int) -> bool:
        if idx < start_frame_idx or idx > end_frame_idx:
            return False
        # Không được nằm trong forbidden_set hay quá gần các frame cấm
        for offset in range(-min_frame_gap, min_frame_gap + 1):
            if (idx + offset) in forbidden_set:
                return False
        return True

    if is_valid(target):
        return target

    # Search mở rộng dần ra 2 phía từ target
    max_offset = end_frame_idx - start_frame_idx
    for offset in range(1, max_offset + 1):
        for candidate in (target + offset, target - offset):
            if is_valid(candidate):
                return candidate

    # Nếu không tìm thấy do min_frame_gap quá nghiêm ngặt, thử relaxed check (chỉ cấm chính xác exact frame)
    for candidate in range(start_frame_idx, end_frame_idx + 1):
        if candidate not in forbidden_set:
            return candidate

    return None
