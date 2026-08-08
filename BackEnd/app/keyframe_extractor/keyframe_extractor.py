"""Class KeyframeExtractor: Entry point chính để trích xuất keyframe bổ sung.

Module 2.4 theo quy định trong ``docs_rule_diagram/module_shot_keyframe (2).md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Set
from pathlib import Path
import time

from BackEnd.app.contracts.pipeline import FrameMetadata, ShotMetadata
from BackEnd.app.keyframe_extractor.config import (
    HybridKeyframeConfig,
    KeyframeSelectionStrategy,
    normalize_strategy,
)
from BackEnd.app.keyframe_extractor.frame_decoder import (
    DEFAULT_MAX_FRAMES_PER_FFMPEG_BATCH,
    extract_and_save_frames,
    extract_and_save_frames_chunked,
)
from BackEnd.app.keyframe_extractor.hybrid_selector import (
    HybridKeyframeSelectionError,
    HybridKeyframeSelector,
)
from BackEnd.app.keyframe_extractor.redundancy import eliminate_cross_shot_duplicates
from BackEnd.app.keyframe_extractor.sampling import (
    DEFAULT_MIN_FRAME_GAP,
    DEFAULT_TARGET_INTERVAL_MS,
    select_additional_keyframe_indices,
)
from BackEnd.app.shot_extractor.video_decoder import probe_fps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "video"
DEFAULT_KEYFRAME_DIR = PROJECT_ROOT / "data" / "keyframes"


class KeyframeExtractor:
    """Trích xuất keyframe bổ sung (`source="extracted"`, `frame_role="keyframe"`) cho từng shot."""

    def __init__(
        self,
        *,
        video_dir: str | Path = DEFAULT_VIDEO_DIR,
        keyframe_dir: str | Path = DEFAULT_KEYFRAME_DIR,
        min_frame_gap: int = DEFAULT_MIN_FRAME_GAP,
        target_interval_ms: int = DEFAULT_TARGET_INTERVAL_MS,
        strategy: KeyframeSelectionStrategy = "time",
        hybrid_config: HybridKeyframeConfig | None = None,
        hybrid_selector: HybridKeyframeSelector | None = None,
        max_frames_per_ffmpeg_batch: int = DEFAULT_MAX_FRAMES_PER_FFMPEG_BATCH,
    ) -> None:
        """Cấu hình KeyframeExtractor.

        Args:
            video_dir: Thư mục chứa các file `<video_id>.mp4`.
            keyframe_dir: Thư mục lưu ảnh keyframe `data/keyframes/<video_id>/`.
            min_frame_gap: Khoảng cách frame tối thiểu không được quá gần keyframe sẵn có.
            target_interval_ms: Mốc thời gian trung bình giữa các keyframe (~2500ms).
            strategy: `"time"` để giữ baseline, `"hybrid_clip"` để chọn theo CLIP có fallback,
                hoặc `"hybrid_clip_strict"` để để lỗi hybrid nổi lên khi debug.
            hybrid_config: Cấu hình thuật toán hybrid.
            hybrid_selector: Dependency injection cho test hoặc custom selector.
            max_frames_per_ffmpeg_batch: Số frame tối đa mỗi batch FFmpeg khi export toàn video.
        """
        if max_frames_per_ffmpeg_batch <= 0:
            raise ValueError("max_frames_per_ffmpeg_batch must be positive.")
        self.video_dir = Path(video_dir)
        self.keyframe_dir = Path(keyframe_dir)
        self.min_frame_gap = min_frame_gap
        self.target_interval_ms = target_interval_ms
        self.strategy = normalize_strategy(strategy)
        self.hybrid_config = hybrid_config or HybridKeyframeConfig(min_frame_gap=min_frame_gap)
        self.hybrid_selector = hybrid_selector
        self.max_frames_per_ffmpeg_batch = max_frames_per_ffmpeg_batch

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

        candidate_indices = self._select_candidate_indices(
            shot,
            video_path=video_path,
            fps=fps,
            existing_frame_idxs=existing_frame_idxs,
        )

        if not candidate_indices:
            return []

        frame_metadatas, output_paths = self._build_frame_metadata(
            shot,
            candidate_indices,
            seq_start=seq_start,
            fps=fps,
        )

        # Thực hiện decode ảnh thật bằng FFmpeg và lưu vào đĩa
        dimensions = extract_and_save_frames(video_path, candidate_indices, output_paths)

        return self._attach_dimensions(frame_metadatas, dimensions)

    def _build_frame_metadata(
        self,
        shot: ShotMetadata,
        candidate_indices: Sequence[int],
        *,
        seq_start: int,
        fps: float,
    ) -> tuple[list[FrameMetadata], list[Path]]:
        """Tạo metadata nháp và output paths cho các frame đã được chọn."""

        output_paths: list[Path] = []
        frame_metadatas: list[FrameMetadata] = []

        video_keyframe_dir = self.keyframe_dir / shot.video_id
        video_keyframe_dir.mkdir(parents=True, exist_ok=True)

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

        return frame_metadatas, output_paths

    @staticmethod
    def _attach_dimensions(
        frame_metadatas: Sequence[FrameMetadata],
        dimensions: Sequence[tuple[int, int]],
    ) -> list[FrameMetadata]:
        """Gắn width/height sau khi ảnh đã được export."""

        if len(frame_metadatas) != len(dimensions):
            raise RuntimeError(
                f"Mismatched extracted metadata and dimensions: "
                f"{len(frame_metadatas)} metadata rows vs {len(dimensions)} dimensions"
            )
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

    def _select_time_candidate_indices(
        self,
        shot: ShotMetadata,
        *,
        fps: float,
        existing_frame_idxs: Sequence[int] | Set[int] | None,
    ) -> list[int]:
        if shot.start_frame_idx is None or shot.end_frame_idx is None:
            raise ValueError(f"Shot '{shot.shot_id}' must have start_frame_idx and end_frame_idx set.")
        return select_additional_keyframe_indices(
            start_frame_idx=shot.start_frame_idx,
            end_frame_idx=shot.end_frame_idx,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            fps=fps,
            existing_frame_idxs=existing_frame_idxs,
            min_frame_gap=self.min_frame_gap,
            target_interval_ms=self.target_interval_ms,
        )

    def _select_candidate_indices(
        self,
        shot: ShotMetadata,
        *,
        video_path: Path,
        fps: float,
        existing_frame_idxs: Sequence[int] | Set[int] | None,
        session: object | None = None,
    ) -> list[int]:
        time_candidates = self._select_time_candidate_indices(
            shot,
            fps=fps,
            existing_frame_idxs=existing_frame_idxs,
        )
        if self.strategy == "time":
            return time_candidates

        selector = self._get_hybrid_selector()
        try:
            try:
                hybrid_candidates = selector.select(
                    shot,
                    video_path=video_path,
                    fps=fps,
                    existing_frame_idxs=existing_frame_idxs,
                    session=session,
                )
            except TypeError:
                hybrid_candidates = selector.select(
                    shot,
                    video_path=video_path,
                    fps=fps,
                    existing_frame_idxs=existing_frame_idxs,
                )
        except HybridKeyframeSelectionError:
            if self.strategy == "hybrid_clip_strict" or not self.hybrid_config.fallback_to_time_sampling:
                raise
            return time_candidates

        if not hybrid_candidates and self.hybrid_config.fallback_to_time_sampling:
            return time_candidates
        return hybrid_candidates

    def _get_hybrid_selector(self) -> HybridKeyframeSelector:
        if self.hybrid_selector is None:
            self.hybrid_selector = HybridKeyframeSelector(config=self.hybrid_config)
        return self.hybrid_selector

    def extract_for_video(
        self,
        video_id: str,
        shots: Sequence[ShotMetadata],
        existing_frame_idxs: Sequence[int] | Set[int] | None = None,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> list[FrameMetadata]:
        """Trích xuất keyframe bổ sung cho toàn bộ các shot của một `video_id`.

        Đảm bảo số `seq` tăng dần liên tục trên toàn bộ video (`_E001`, `_E002`, ...).
        """
        video_path = self._resolve_video_path(video_id)
        fps = probe_fps(video_path)

        all_candidate_indices: list[int] = []
        all_output_paths: list[Path] = []
        all_frame_metadatas: list[FrameMetadata] = []
        current_seq = 1

        # Cập nhật existing_frame_idxs liên tục khi trích xuất
        all_existing: set[int] = set(existing_frame_idxs) if existing_frame_idxs is not None else set()

        sorted_shots = sorted(shots, key=lambda s: s.shot_index)
        total_shots = len(sorted_shots)

        session = None
        if self.strategy != "time":
            try:
                from BackEnd.app.embedding.clip.video_session import PyAVVideoSession
                from BackEnd.app.keyframe_extractor.candidate_sampler import (
                    sample_candidate_frame_indices,
                )

                session = PyAVVideoSession(video_path, fps)

                # Thu thập toàn bộ candidate timestamps của các shot để Single-Pass Prefetch trong 10s
                all_video_timestamps: list[int] = []
                for shot in sorted_shots:
                    if shot.start_frame_idx is not None and shot.end_frame_idx is not None:
                        c_idxs = sample_candidate_frame_indices(
                            start_frame_idx=shot.start_frame_idx,
                            end_frame_idx=shot.end_frame_idx,
                            fps=fps,
                            sample_fps=self.hybrid_config.sample_fps,
                            min_frame_gap=self.hybrid_config.min_frame_gap,
                            transition_margin_frames=self.hybrid_config.transition_margin_frames,
                            max_candidate_frames_per_shot=self.hybrid_config.max_candidate_frames_per_shot,
                        )
                        all_video_timestamps.extend([round(idx / fps * 1000) for idx in c_idxs])

                if hasattr(session, "prefetch_timestamps"):
                    session.prefetch_timestamps(all_video_timestamps)
            except Exception:
                session = None

        try:
            per_shot_records: list[tuple[ShotMetadata, list[object]]] = []
            for shot_number, shot in enumerate(sorted_shots, start=1):
                if progress_callback is not None:
                    progress_callback(
                        self._build_progress_event(
                            shot,
                            phase="start",
                            shot_number=shot_number,
                            total_shots=total_shots,
                            selected_count=0,
                            elapsed_s=0.0,
                        )
                    )
                selection_start = time.perf_counter()
                candidate_indices = self._select_candidate_indices(
                    shot,
                    video_path=video_path,
                    fps=fps,
                    existing_frame_idxs=all_existing,
                    session=session,
                )
                
                shot_cands = []
                if self.hybrid_selector is not None and hasattr(self.hybrid_selector, "last_redundancy_candidates"):
                    shot_cands = list(getattr(self.hybrid_selector, "last_redundancy_candidates", []))

                per_shot_records.append((shot, shot_cands if shot_cands else candidate_indices))

                if progress_callback is not None:
                    progress_callback(
                        self._build_progress_event(
                            shot,
                            phase="done",
                            shot_number=shot_number,
                            total_shots=total_shots,
                            selected_count=len(candidate_indices),
                            elapsed_s=time.perf_counter() - selection_start,
                        )
                    )

            # Hậu xử lý Cross-Shot Dedup Filter
            if per_shot_records and any(rec[1] and hasattr(rec[1][0], "clip_vector") for rec in per_shot_records):
                cands_list = [rec[1] for rec in per_shot_records]
                filtered_per_shot = eliminate_cross_shot_duplicates(cands_list)
            else:
                filtered_per_shot = [
                    [c.frame_idx if hasattr(c, "frame_idx") else c for c in rec[1]]
                    for rec in per_shot_records
                ]

            for (shot, _), candidate_indices in zip(per_shot_records, filtered_per_shot):
                if candidate_indices:
                    frame_metadatas, output_paths = self._build_frame_metadata(
                        shot,
                        candidate_indices,
                        seq_start=current_seq,
                        fps=fps,
                    )
                    all_candidate_indices.extend(candidate_indices)
                    all_output_paths.extend(output_paths)
                    all_frame_metadatas.extend(frame_metadatas)
                    all_existing.update(candidate_indices)
                    current_seq += len(candidate_indices)

            if not all_candidate_indices:
                return []

            export_start = time.perf_counter()
            dimensions = extract_and_save_frames_chunked(
                video_path,
                all_candidate_indices,
                all_output_paths,
                chunk_size=self.max_frames_per_ffmpeg_batch,
            )
            export_s = time.perf_counter() - export_start

            if progress_callback is not None:
                progress_callback(
                    {
                        "video_id": video_id,
                        "phase": "export",
                        "strategy": self.strategy,
                        "total_shots": total_shots,
                        "frame_count": len(all_candidate_indices),
                        "chunk_size": self.max_frames_per_ffmpeg_batch,
                        "chunk_count": (
                            len(all_candidate_indices) + self.max_frames_per_ffmpeg_batch - 1
                        )
                        // self.max_frames_per_ffmpeg_batch,
                        "export_s": export_s,
                    }
                )

            return self._attach_dimensions(all_frame_metadatas, dimensions)
        finally:
            if session is not None and hasattr(session, "close"):
                session.close()

    def _build_progress_event(
        self,
        shot: ShotMetadata,
        *,
        phase: str,
        shot_number: int,
        total_shots: int,
        selected_count: int,
        elapsed_s: float,
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "video_id": shot.video_id,
            "shot_id": shot.shot_id,
            "shot_index": shot.shot_index,
            "shot_number": shot_number,
            "total_shots": total_shots,
            "phase": phase,
            "strategy": self.strategy,
            "selected_count": selected_count,
            "elapsed_s": elapsed_s,
        }
        if self.strategy != "time" and self.hybrid_selector is not None:
            event["hybrid"] = dict(getattr(self.hybrid_selector, "last_metrics", {}) or {})
        return event
