"""Logic thuần cho ranh giới shot: predictions của model -> ``ShotMetadata`` đã validate.

Mọi thứ trong module này đều là Python/NumPy thuần, không phụ thuộc FFmpeg,
PyTorch, hay filesystem, nên có thể unit-test độc lập với pipeline inference
(vốn chậm và cần GPU). ``shot_extractor.py`` là nơi duy nhất gọi module này.

Có hai quyết định độc lập diễn ra ở đây, cả hai đều được ghi lại trong
``Markdown_Doc/module_shot_keyframe.md`` mục 2.3:

1. ``predictions_to_scene_frames`` biến xác suất ranh giới của từng frame
   thành các khoảng frame theo shot, dùng threshold 0.5 — port trực tiếp từ
   hàm ``predictions_to_scenes`` gốc của TransNetV2 để giữ đúng ngữ nghĩa
   ranh giới mà threshold này được chọn theo (mục 4 trong tài liệu pipeline).
2. ``scene_frames_to_shots`` quy đổi các khoảng frame đó sang mốc mili-giây
   và gộp các shot ngắn hơn 500ms vào shot lân cận, coi chúng là nhiễu của
   detector chứ không phải một lần cắt cảnh có ý nghĩa.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from BackEnd.app.contracts.pipeline import ShotMetadata

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SHOT_DURATION_MS = 500


def predictions_to_scene_frames(
    predictions: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> list[tuple[int, int]]:
    """Quy đổi điểm ranh giới của từng frame thành các khoảng frame theo shot (inclusive).

    Port từ hàm ``predictions_to_scenes`` gốc của TransNetV2 (xem
    ``inference/transnetv2.py`` trong soCzech/TransNetV2) để giữ đúng ngữ
    nghĩa ranh giới: một frame được coi là "đang chuyển cảnh" khi điểm của
    nó vượt ``threshold``, và một shot kết thúc ở frame ngay trước khi đợt
    chuyển cảnh bắt đầu, rồi bắt đầu lại ở frame mà đợt chuyển cảnh đó kết
    thúc. Khi một đợt chuyển cảnh kéo dài nhiều frame liên tiếp, các frame
    ở giữa đợt đó không thuộc shot nào cả và bị loại có chủ đích khỏi mọi
    khoảng trả về — đây là hành vi gốc đã biết trước, không phải bug (xem
    ``Markdown_Doc/shot_extractor_pipeline.md``).

    Trả về danh sách các cặp ``(start_frame_idx, end_frame_idx)``, cả hai
    đều inclusive và đánh số từ 0, theo thứ tự tăng dần và không chồng lấn.
    """

    if predictions.ndim != 1:
        raise ValueError(f"predictions must be 1-D, got shape {predictions.shape}")
    if len(predictions) == 0:
        raise ValueError("predictions must not be empty")

    is_transition = (predictions > threshold).astype(np.uint8)

    scenes: list[tuple[int, int]] = []
    prev_state = 0
    start = 0
    state = 0
    for index, state in enumerate(is_transition):
        state = int(state)
        if prev_state == 1 and state == 0:
            start = index
        if prev_state == 0 and state == 1 and index != 0:
            scenes.append((start, index))
        prev_state = state

    if state == 0:
        scenes.append((start, len(predictions) - 1))

    if not scenes:
        # Mọi frame đều bị chấm điểm "đang chuyển cảnh": coi cả video là một
        # shot duy nhất thay vì trả về danh sách rỗng.
        return [(0, len(predictions) - 1)]
    return scenes


def scene_frames_to_shots(
    scene_frames: Sequence[tuple[int, int]],
    *,
    video_id: str,
    fps: float,
    min_shot_duration_ms: int = DEFAULT_MIN_SHOT_DURATION_MS,
) -> list[ShotMetadata]:
    """Xây dựng danh sách ``ShotMetadata`` đã validate, có thứ tự, từ các khoảng scene frame thô.

    Quy ước thời gian: khoảng của mỗi shot là nửa-mở ``[start_ms, end_ms)``.
    ``start_ms`` là thời điểm hiển thị của ``start_frame_idx``; ``end_ms`` là
    thời điểm hiển thị của frame *ngay sau* ``end_frame_idx`` (tức thời điểm
    frame cuối cùng của shot ngừng được hiển thị), nhờ đó luôn đảm bảo
    ``end_ms > start_ms`` kể cả với shot chỉ có 1 frame.

    Các shot ngắn hơn ``min_shot_duration_ms`` được coi là nhiễu của detector
    và được gộp vào một shot lân cận thay vì giữ làm record riêng (xem
    docstring của module). ``shot_index`` được đánh lại tuần tự từ 0 sau khi
    gộp xong.
    """

    if fps <= 0:
        raise ValueError(f"fps must be positive for video '{video_id}', got {fps}")
    if not scene_frames:
        raise ValueError(f"No scene boundaries to convert for video '{video_id}'")

    raw_shots: list[list[int]] = []
    previous_end_frame_idx = -1
    for start_frame_idx, end_frame_idx in scene_frames:
        if end_frame_idx < start_frame_idx:
            raise ValueError(
                f"Invalid scene range for video '{video_id}': "
                f"end_frame_idx {end_frame_idx} < start_frame_idx {start_frame_idx}"
            )
        if start_frame_idx <= previous_end_frame_idx:
            raise ValueError(
                f"Scene ranges for video '{video_id}' are not strictly ordered: "
                f"start_frame_idx {start_frame_idx} does not come after the "
                f"previous shot's end_frame_idx {previous_end_frame_idx}"
            )
        previous_end_frame_idx = end_frame_idx

        start_ms = round(start_frame_idx / fps * 1000)
        end_ms = round((end_frame_idx + 1) / fps * 1000)
        if end_ms <= start_ms:
            # Chỉ có thể xảy ra do làm tròn cực đoan với input fps rất cao;
            # vẫn phải đảm bảo bất biến `end_ms > start_ms` mà DB yêu cầu.
            end_ms = start_ms + 1
        raw_shots.append([start_frame_idx, end_frame_idx, start_ms, end_ms])

    merged = _merge_short_shots(raw_shots, min_shot_duration_ms)

    shots = [
        ShotMetadata(
            shot_id=f"{video_id}_S{shot_index:03d}",
            video_id=video_id,
            shot_index=shot_index,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
        for shot_index, (start_frame_idx, end_frame_idx, start_ms, end_ms) in enumerate(merged)
    ]
    _validate_shots(shots, video_id=video_id)
    return shots


def _merge_short_shots(
    raw_shots: list[list[int]], min_shot_duration_ms: int
) -> list[list[int]]:
    """Nuốt các shot ngắn hơn ``min_shot_duration_ms`` vào một shot lân cận.

    Một shot ngắn được gộp vào shot liền trước (đã được emit) bằng cách mở
    rộng ranh giới kết thúc của shot đó. Trường hợp duy nhất không có shot
    liền trước — shot đầu tiên của video bị ngắn — được gộp xuôi vào shot kế
    tiếp thay vào đó, để một phát hiện nhiễu ở đầu video không bao giờ trở
    thành "mồ côi" không gộp được vào đâu.
    """

    if len(raw_shots) <= 1:
        return raw_shots

    merged: list[list[int]] = []
    for shot in raw_shots:
        _, _, start_ms, end_ms = shot
        duration_ms = end_ms - start_ms
        if duration_ms < min_shot_duration_ms and merged:
            previous = merged[-1]
            previous[1] = shot[1]  # mở rộng end_frame_idx
            previous[3] = shot[3]  # mở rộng end_ms
        else:
            merged.append(list(shot))

    first_start_ms, first_end_ms = merged[0][2], merged[0][3]
    if len(merged) > 1 and (first_end_ms - first_start_ms) < min_shot_duration_ms:
        first = merged.pop(0)
        merged[0][0] = first[0]  # kéo start_frame_idx lùi về điểm bắt đầu của shot mồ côi
        merged[0][2] = first[2]  # kéo start_ms lùi về điểm bắt đầu của shot mồ côi

    return merged


def _validate_shots(shots: list[ShotMetadata], *, video_id: str) -> None:
    """Kiểm tra các bất biến mà schema database yêu cầu (fail nhanh, kèm ngữ cảnh)."""

    previous_end_ms = -1
    previous_end_frame_idx = -1
    for shot in shots:
        if shot.start_ms < 0:
            raise ValueError(f"{shot.shot_id}: start_ms {shot.start_ms} < 0")
        if shot.end_ms <= shot.start_ms:
            raise ValueError(f"{shot.shot_id}: end_ms {shot.end_ms} <= start_ms {shot.start_ms}")
        if shot.start_frame_idx is None or shot.end_frame_idx is None:
            raise ValueError(f"{shot.shot_id}: start_frame_idx/end_frame_idx must be set")
        if shot.start_frame_idx < 0:
            raise ValueError(f"{shot.shot_id}: start_frame_idx {shot.start_frame_idx} < 0")
        if shot.end_frame_idx < shot.start_frame_idx:
            raise ValueError(
                f"{shot.shot_id}: end_frame_idx {shot.end_frame_idx} < "
                f"start_frame_idx {shot.start_frame_idx}"
            )
        if shot.start_ms < previous_end_ms or shot.start_frame_idx <= previous_end_frame_idx:
            raise ValueError(
                f"Shots for video '{video_id}' overlap or are out of order at {shot.shot_id}"
            )
        previous_end_ms = shot.end_ms
        previous_end_frame_idx = shot.end_frame_idx
