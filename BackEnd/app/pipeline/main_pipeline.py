"""Memory-conscious orchestration for the offline preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable
import gc
from typing import Any

from BackEnd.CONFIG import (
    CLIP_DIMENSION,
    CLIP_MODEL,
    PIPELINE_MAX_OOM_RETRIES,
    PIPELINE_PARALLEL_MODE,
)
from BackEnd.app.clip_extractor import ClipExtractor
from BackEnd.app.contracts.pipeline import VideoMetadata
from BackEnd.app.database.faiss_db import FAISS_Manager
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.keyframe_extractor import KeyframeExtractor
from BackEnd.app.ocr.service import OCRService
from BackEnd.app.pipeline import (
    embedding,
    extract_clip,
    extract_keyframe,
    extract_shot,
    ocr,
    tracking,
)
from BackEnd.app.pipeline.embedding import EmbeddingPipeline
from BackEnd.app.pipeline.parallel_runner import (
    GPUPreflightError,
    WorkerResult,
    check_parallel_gpu_preflight,
    run_parallel_workers,
    run_worker_exclusively,
)
from BackEnd.app.shot_extractor import ShotExtractor
from BackEnd.app.tracking.tracking import YOLOTrackingService


class Pipeline:
    """Run dependency-safe offline stages with optional bounded GPU overlap."""

    def __init__(
        self,
        *,
        db: PostgreManager,
        faiss: FAISS_Manager,
        videos: list[VideoMetadata] | None = None,
        parallel_mode: str | None = None,
        shot_extractor_factory: Callable[[], ShotExtractor] = ShotExtractor,
        keyframe_extractor_factory: Callable[[], KeyframeExtractor] = KeyframeExtractor,
        clip_extractor_factory: Callable[[], ClipExtractor] = ClipExtractor,
        ocr_service_factory: Callable[[], OCRService] = OCRService,
        tracker_factory: Callable[[], YOLOTrackingService] = YOLOTrackingService,
        embedding_pipeline_factory: Callable[
            [PostgreManager, FAISS_Manager], EmbeddingPipeline
        ] | None = None,
    ) -> None:
        self.db = db
        self.faiss = faiss
        self.videos = list(videos) if videos is not None else db.get_list_video()
        self._parallel_mode = parallel_mode or PIPELINE_PARALLEL_MODE
        self._shot_extractor_factory = shot_extractor_factory
        self._keyframe_extractor_factory = keyframe_extractor_factory
        self._clip_extractor_factory = clip_extractor_factory
        self._ocr_service_factory = ocr_service_factory
        self._tracker_factory = tracker_factory
        self._embedding_pipeline_factory = (
            embedding_pipeline_factory or _build_embedding_pipeline
        )

    def run(self) -> None:
        """Run all wired stages, overlapping tracking and OCR when safe."""

        self.run_shot_extraction()
        self.run_dependent_stages()
        self.run_embeddings()

    def run_dependent_stages(self) -> None:
        """Run tracking beside keyframe/clip/OCR, or use the safe fallback."""

        mode = self._parallel_mode.lower()
        if mode not in {"auto", "parallel", "sequential"}:
            raise ValueError(
                "PIPELINE_PARALLEL_MODE must be 'auto', 'parallel' or 'sequential'."
            )
        if mode == "sequential":
            self._run_dependent_stages_sequentially()
            return

        if not self._uses_default_process_workers():
            message = (
                "Parallel mode requires the default stage factories because "
                "custom factories cannot be sent safely to spawned workers."
            )
            if mode == "parallel":
                raise GPUPreflightError(message)
            self._run_dependent_stages_sequentially()
            return

        preflight = check_parallel_gpu_preflight()
        if not preflight.parallel_safe:
            if mode == "parallel":
                raise GPUPreflightError(preflight.reason)
            self._run_dependent_stages_sequentially()
            return

        video_ids = tuple(video.video_id for video in self.videos)
        if not video_ids:
            return
        tracking_result, enrichment_result = run_parallel_workers(
            video_ids,
            tuple(reversed(video_ids)),
        )
        self._retry_oom_result(tracking_result)
        self._retry_oom_result(enrichment_result)

    def _run_dependent_stages_sequentially(self) -> None:
        self.run_keyframe_extraction()
        self.run_clip_extraction()
        self.run_ocr()
        self.run_tracking()

    def _retry_oom_result(self, result: WorkerResult) -> None:
        """Retry only unfinished videos after the concurrent CUDA contexts exit."""

        current = result
        retries = 0
        while current.oom_video_id is not None:
            if retries >= PIPELINE_MAX_OOM_RETRIES:
                raise RuntimeError(
                    f"{current.kind} exhausted {PIPELINE_MAX_OOM_RETRIES} OOM "
                    f"retries at video '{current.oom_video_id}': "
                    f"{current.oom_message}"
                )
            completed = set(current.completed_video_ids)
            pending = tuple(
                video_id
                for video_id in current.requested_video_ids
                if video_id not in completed
            )
            if not pending:
                return
            retries += 1
            current = run_worker_exclusively(
                current.kind,
                pending,
                ocr_retry_level=retries if current.kind == "enrichment" else 0,
            )

    def _uses_default_process_workers(self) -> bool:
        return (
            self._keyframe_extractor_factory is KeyframeExtractor
            and self._clip_extractor_factory is ClipExtractor
            and self._ocr_service_factory is OCRService
            and self._tracker_factory is YOLOTrackingService
        )

    def run_shot_extraction(self) -> None:
        extractor = self._shot_extractor_factory()
        try:
            for video in self.videos:
                extract_shot.extract_shot(video.video_id, self.db, extractor)
        finally:
            _release_stage_resource(extractor)

    def run_keyframe_extraction(self) -> None:
        extractor = self._keyframe_extractor_factory()
        try:
            for video in self.videos:
                extract_keyframe.extract_keyframes(video.video_id, self.db, extractor)
        finally:
            _release_stage_resource(extractor)

    def run_clip_extraction(self) -> None:
        extractor = self._clip_extractor_factory()
        try:
            for video in self.videos:
                extract_clip.extract_clip(video.video_id, self.db, extractor)
        finally:
            _release_stage_resource(extractor)

    def run_ocr(self) -> None:
        service = self._ocr_service_factory()
        try:
            for video in self.videos:
                ocr.run_ocr(video.video_id, self.db, service)
        finally:
            _release_stage_resource(service)

    def run_tracking(self) -> None:
        tracker = self._tracker_factory()
        try:
            for video in self.videos:
                tracking.track_video(video, self.db, tracker)
        finally:
            _release_stage_resource(tracker)

    def run_embeddings(self) -> None:
        pipeline = self._embedding_pipeline_factory(self.db, self.faiss)
        try:
            for video in self.videos:
                embedding.embed_frames(video.video_id, pipeline)
            for video in self.videos:
                embedding.embed_clips(video.video_id, pipeline)
            for video in self.videos:
                embedding.embed_shots(video.video_id, pipeline)
        finally:
            _release_stage_resource(pipeline)


def _build_embedding_pipeline(
    db: PostgreManager,
    faiss: FAISS_Manager,
) -> EmbeddingPipeline:
    return EmbeddingPipeline(db=db, faiss_manager=faiss)


def _release_stage_resource(resource: Any) -> None:
    """Release a stage resource once after its complete video batch."""

    close = getattr(resource, "close", None)
    if callable(close):
        close()

    del resource
    gc.collect()

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run the complete pipeline for every video stored in PostgreSQL."""

    db = PostgreManager()
    try:
        faiss = FAISS_Manager(
            img_dim=CLIP_DIMENSION,
            clip_dim=CLIP_DIMENSION,
            shot_dim=CLIP_DIMENSION,
            model_name=CLIP_MODEL,
            data_path="artifacts/faiss",
            load_existing=True,
        )
        videos = db.get_list_video()
        Pipeline(db=db, faiss=faiss, videos=videos).run()
        print(f"Pipeline completed for all {len(videos)} video(s).")
    finally:
        db.engine.dispose()


__all__ = ["Pipeline"]


if __name__ == "__main__":
    main()
