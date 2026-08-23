"""Hybrid CLIP-based keyframe selector."""

from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path
import time

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.candidate_decoder import PyAVCandidateFrameDecoder
from BackEnd.app.keyframe_extractor.candidate_sampler import sample_candidate_frame_indices
from BackEnd.app.keyframe_extractor.clip_adapter import ClipImageEmbeddingAdapter, ImageEmbeddingAdapter
from BackEnd.app.keyframe_extractor.config import HybridKeyframeConfig
from BackEnd.app.keyframe_extractor.facility_selection import select_facility_representatives
from BackEnd.app.keyframe_extractor.redundancy import (
    RedundancyCandidate,
    eliminate_redundant_candidates,
)


class HybridKeyframeSelectionError(RuntimeError):
    """Raised when hybrid semantic keyframe selection cannot complete."""


class HybridKeyframeSelector:
    """Select keyframes using sparse sampling, CLIP coverage, and deduplication."""

    def __init__(
        self,
        *,
        config: HybridKeyframeConfig | None = None,
        decoder: object | None = None,
        embedding_adapter: ImageEmbeddingAdapter | None = None,
    ) -> None:
        self.config = config or HybridKeyframeConfig()
        self.decoder = decoder or PyAVCandidateFrameDecoder()
        self.embedding_adapter = embedding_adapter or ClipImageEmbeddingAdapter()
        self.last_metrics: dict[str, int | float | str] = {}

    def select(
        self,
        shot: ShotMetadata,
        *,
        video_path: str | Path,
        fps: float,
        existing_frame_idxs: Sequence[int] | Set[int] | None = None,
        session: object | None = None,
    ) -> list[int]:
        """Return final hybrid-selected frame indices for one shot."""

        total_start = time.perf_counter()
        self.last_redundancy_candidates: list[RedundancyCandidate] = []
        self.last_metrics = {
            "candidate_count": 0,
            "reference_count": 0,
            "selected_count": 0,
            "sample_s": 0.0,
            "decode_s": 0.0,
            "clip_s": 0.0,
            "selection_s": 0.0,
            "redundancy_s": 0.0,
            "total_s": 0.0,
            "status": "started",
        }
        if shot.start_frame_idx is None or shot.end_frame_idx is None:
            raise ValueError(f"Shot '{shot.shot_id}' must have frame bounds.")

        sample_start = time.perf_counter()
        candidates = sample_candidate_frame_indices(
            start_frame_idx=shot.start_frame_idx,
            end_frame_idx=shot.end_frame_idx,
            fps=fps,
            sample_fps=self.config.sample_fps,
            existing_frame_idxs=existing_frame_idxs,
            min_frame_gap=self.config.min_frame_gap,
            transition_margin_frames=self.config.transition_margin_frames,
            max_candidate_frames_per_shot=self.config.max_candidate_frames_per_shot,
        )
        self.last_metrics["sample_s"] = time.perf_counter() - sample_start
        self.last_metrics["candidate_count"] = len(candidates)
        if not candidates:
            self.last_metrics["status"] = "no_candidates"
            self.last_metrics["total_s"] = time.perf_counter() - total_start
            return []

        try:
            existing_in_shot = _evenly_limit_indices(
                sorted(
                    frame_idx
                    for frame_idx in set(existing_frame_idxs or [])
                    if shot.start_frame_idx <= frame_idx <= shot.end_frame_idx
                ),
                self.config.max_reference_frames_per_shot,
            )
            self.last_metrics["reference_count"] = len(existing_in_shot)
            # Decode the existing official/extracted frames only for semantic
            # coverage. They are never returned as new candidates.
            scoring_indices = sorted(set(candidates) | set(existing_in_shot))
            decode_start = time.perf_counter()
            try:
                images_by_frame = self.decoder.decode(
                    video_id=shot.video_id,
                    video_path=video_path,
                    frame_indices=scoring_indices,
                    fps=fps,
                    session=session,
                )
            except TypeError:
                images_by_frame = self.decoder.decode(
                    video_id=shot.video_id,
                    video_path=video_path,
                    frame_indices=scoring_indices,
                    fps=fps,
                )
            self.last_metrics["decode_s"] = time.perf_counter() - decode_start
            images = [images_by_frame[frame_idx] for frame_idx in scoring_indices]
            clip_start = time.perf_counter()
            scoring_vectors = self.embedding_adapter.encode_images(images)
            self.last_metrics["clip_s"] = time.perf_counter() - clip_start
            vector_by_frame = {
                frame_idx: scoring_vectors[row] for row, frame_idx in enumerate(scoring_indices)
            }
            candidate_vectors = [vector_by_frame[frame_idx] for frame_idx in candidates]
            reference_vectors = [vector_by_frame[frame_idx] for frame_idx in existing_in_shot]
            selection_start = time.perf_counter()
            selected_indices = select_facility_representatives(
                candidates,
                candidate_vectors,
                max_representatives=self.config.max_additional_per_shot,
                reference_vectors=reference_vectors or None,
                min_marginal_gain=self.config.min_marginal_gain,
            )
            self.last_metrics["selection_s"] = time.perf_counter() - selection_start
            semantic_candidates = [
                RedundancyCandidate(
                    frame_idx=frame_idx,
                    image=images_by_frame[frame_idx],
                    clip_vector=vector_by_frame[frame_idx],
                )
                for frame_idx in selected_indices
            ]
            redundancy_start = time.perf_counter()
            selected = eliminate_redundant_candidates(
                semantic_candidates,
                existing_frame_idxs=existing_frame_idxs,
                max_output=self.config.max_additional_per_shot,
                hsv_similarity_threshold=self.config.hsv_similarity_threshold,
                clip_similarity_threshold=self.config.clip_similarity_threshold,
                low_information_min_nonzero_bins=self.config.low_information_min_nonzero_bins,
            )
            self.last_metrics["redundancy_s"] = time.perf_counter() - redundancy_start
            self.last_redundancy_candidates = [
                c for c in semantic_candidates if c.frame_idx in selected
            ]
            self.last_metrics["selected_count"] = len(selected)
            self.last_metrics["status"] = "success"
            self.last_metrics["total_s"] = time.perf_counter() - total_start
            return selected
        except Exception as error:
            self.last_metrics["status"] = "error"
            self.last_metrics["total_s"] = time.perf_counter() - total_start
            raise HybridKeyframeSelectionError(
                f"Hybrid keyframe selection failed for shot '{shot.shot_id}'."
            ) from error


def _evenly_limit_indices(indices: Sequence[int], limit: int) -> list[int]:
    """Keep temporal coverage while bounding official-frame decode and CLIP work."""

    if len(indices) <= limit:
        return list(indices)
    if limit == 1:
        return [indices[len(indices) // 2]]
    positions = [round(index * (len(indices) - 1) / (limit - 1)) for index in range(limit)]
    return [indices[position] for position in positions]
