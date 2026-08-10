"""Class KeyframeExtractor: Entry point chính để trích xuất keyframe bổ sung.

Module 2.4 theo quy định trong ``docs_rule_diagram/module_shot_keyframe (2).md``.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path

from BackEnd.CONFIG import (
    KEYFRAME_OUTPUT_DIR as DEFAULT_KEYFRAME_DIR,
    PROJECT_ROOT,
    VIDEO_DIR as DEFAULT_VIDEO_DIR,
)
from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata
from BackEnd.app.keyframe_extractor.frame_decoder import extract_and_save_frames
from BackEnd.app.keyframe_extractor.sampling import (
    DEFAULT_MIN_FRAME_GAP,
    DEFAULT_TARGET_INTERVAL_MS,
    select_additional_keyframe_indices,
)
from BackEnd.app.shot_extractor.video_decoder import probe_fps

class KeyframeExtractor:
    """Trích xuất keyframe bổ sung (`source="extracted"`, `frame_role="keyframe"`) cho từng shot."""

    def __init__(
        self,
        *,
        video_dir: str | Path = DEFAULT_VIDEO_DIR,
        keyframe_dir: str | Path = DEFAULT_KEYFRAME_DIR,
        min_frame_gap: int = DEFAULT_MIN_FRAME_GAP,
        target_interval_ms: int = DEFAULT_TARGET_INTERVAL_MS,
    ) -> None:
        """Cấu hình KeyframeExtractor.

        Args:
            video_dir: Thư mục chứa các file `<video_id>.mp4`.
            keyframe_dir: Thư mục lưu ảnh keyframe `data/keyframes/<video_id>/`.
            min_frame_gap: Khoảng cách frame tối thiểu không được quá gần keyframe sẵn có.
            target_interval_ms: Mốc thời gian trung bình giữa các keyframe (~2500ms).
        """
        self.video_dir = Path(video_dir)
        self.keyframe_dir = Path(keyframe_dir)
        self.min_frame_gap = min_frame_gap
        self.target_interval_ms = target_interval_ms

    def _resolve_video_path(self, video_id: str) -> Path:
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.is_file():
            candidates = sorted(PROJECT_ROOT.glob(f"data/**/{video_id}.mp4"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"Video not found for '{video_id}': {video_path}")
        return video_path

    def extract(
        self,
        shot: ShotMetadata,
        existing_frame_idxs: Sequence[int] | Set[int] | None = None,
        *,
        seq_start: int = 1,
        fps: float | None = None,
    ) -> list[FrameMetadata]:
        """Trích xuất keyframe bổ sung cho 1 `shot` đơn lẻ."""
        if shot.start_frame_idx is None or shot.end_frame_idx is None:
            raise ValueError(f"Shot '{shot.shot_id}' must have start_frame_idx and end_frame_idx set.")

        video_path = self._resolve_video_path(shot.video_id)
        if fps is None or fps <= 0:
            fps = probe_fps(video_path)

        candidate_indices = select_additional_keyframe_indices(
            start_frame_idx=shot.start_frame_idx,
            end_frame_idx=shot.end_frame_idx,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            fps=fps,
            existing_frame_idxs=existing_frame_idxs,
            min_frame_gap=self.min_frame_gap,
            target_interval_ms=self.target_interval_ms,
        )

        if not candidate_indices:
            return []

        # Tạo đường dẫn lưu ảnh và sinh FrameMetadata
        output_paths: list[Path] = []
        frame_metadatas: list[FrameMetadata] = []

        video_keyframe_dir = self.keyframe_dir / shot.video_id

        for idx_offset, candidate_idx in enumerate(candidate_indices):
            seq = seq_start + idx_offset
            frame_id = f"{shot.video_id}_E{seq:03d}"
            output_path = video_keyframe_dir / f"{frame_id}.jpg"
            output_paths.append(output_path)

            timestamp_ms = round(candidate_idx / fps * 1000)

            frame_metadata = FrameMetadata(
                frame_id=frame_id,
                video_id=shot.video_id,
                shot_id=shot.shot_id,
                timestamp_ms=timestamp_ms,
                fps=fps,
                frame_idx=candidate_idx,
                frame_role="keyframe",
                source="extracted",
                n=candidate_idx,
                pts_time=None,
                frame_path=output_path,
                width=None,   # Sẽ được cập nhật sau khi decode
                height=None,  # Sẽ được cập nhật sau khi decode
            )
            frame_metadatas.append(frame_metadata)

        # Thực hiện decode ảnh thật bằng FFmpeg và lưu vào đĩa
        dimensions = extract_and_save_frames(video_path, candidate_indices, output_paths)

        # Cập nhật width, height vào result metadata
        results: list[FrameMetadata] = []
        for meta, (w, h) in zip(frame_metadatas, dimensions):
            updated_meta = FrameMetadata(
                frame_id=meta.frame_id,
                video_id=meta.video_id,
                shot_id=meta.shot_id,
                timestamp_ms=meta.timestamp_ms,
                fps=meta.fps,
                frame_idx=meta.frame_idx,
                frame_role=meta.frame_role,
                source=meta.source,
                n=meta.n,
                pts_time=meta.pts_time,
                frame_path=meta.frame_path,
                width=w,
                height=h,
            )
            results.append(updated_meta)

        return results

    def extract_for_video(
        self,
        video_id: str,
        shots: Sequence[ShotMetadata],
        existing_frame_idxs: Sequence[int] | Set[int] | None = None,
    ) -> list[FrameMetadata]:
        """Trích xuất keyframe bổ sung cho toàn bộ các shot của một `video_id`.

        Đảm bảo số `seq` tăng dần liên tục trên toàn bộ video (`_E001`, `_E002`, ...).
        """
        video_path = self._resolve_video_path(video_id)
        fps = probe_fps(video_path)

        all_extracted: list[FrameMetadata] = []
        current_seq = 1

        # Cập nhật existing_frame_idxs liên tục khi trích xuất
        all_existing: set[int] = set(existing_frame_idxs) if existing_frame_idxs is not None else set()

        for shot in sorted(shots, key=lambda s: s.shot_index):
            extracted = self.extract(
                shot,
                existing_frame_idxs=all_existing,
                seq_start=current_seq,
                fps=fps,
            )
            for item in extracted:
                all_extracted.append(item)
                all_existing.add(item.frame_idx)
            current_seq += len(extracted)

        return all_extracted
