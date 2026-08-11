"""Memory-conscious orchestration for the offline preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable
import gc
from typing import Any

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
from BackEnd.app.shot_extractor import ShotExtractor
from BackEnd.app.tracking.tracking import YOLOTrackingService


class Pipeline:
    """Run one GPU-heavy stage at a time across the selected videos.

    Factories defer model construction until the stage actually starts.  A
    service is reused for every video in its stage and released before the
    next heavyweight stage begins.
    """

    def __init__(
        self,
        *,
        db: PostgreManager,
        faiss: FAISS_Manager,
        videos: list[VideoMetadata] | None = None,
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
        self._shot_extractor_factory = shot_extractor_factory
        self._keyframe_extractor_factory = keyframe_extractor_factory
        self._clip_extractor_factory = clip_extractor_factory
        self._ocr_service_factory = ocr_service_factory
        self._tracker_factory = tracker_factory
        self._embedding_pipeline_factory = (
            embedding_pipeline_factory or _build_embedding_pipeline
        )

    def run(self) -> None:
        """Run all currently wired pipeline stages in dependency order."""

        self.run_shot_extraction()
        self.run_keyframe_extraction()
        self.run_clip_extraction()
        self.run_ocr()
        self.run_tracking()
        self.run_embeddings()

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


__all__ = ["Pipeline"]
