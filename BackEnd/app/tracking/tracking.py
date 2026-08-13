"""Track COCO objects with YOLO26 while decoding each video only once."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Iterator

import av
import numpy as np

from BackEnd.CONFIG import PROJECT_ROOT, TrackingConfig
from BackEnd.app.contracts.pipeline import (
    ObjectTrackResult,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
)
from BackEnd.app.tracking.class_mapping import (
    COCO_TO_OPENIMAGES,
    get_canonical_class,
    validate_coco_names,
)


@dataclass(frozen=True, slots=True)
class TrackingBatchResult:
    """Track summaries and independent YOLO observations for one video."""

    tracks: list[ObjectTrackResult]
    observations: list[TrackObservationResult]


@dataclass(slots=True)
class _TrackAccumulator:
    track_id: int
    shot_id: str
    class_id: str
    start_ms: int
    end_ms: int
    start_frame_idx: int
    end_frame_idx: int
    confidence_sum: float = 0.0
    observation_count: int = 0

    def add(self, timestamp_ms: int, frame_idx: int, confidence: float) -> None:
        self.end_ms = timestamp_ms
        self.end_frame_idx = frame_idx
        self.confidence_sum += confidence
        self.observation_count += 1


def _iter_video_frames(video_path: Path) -> Iterator[tuple[int, int, Any]]:
    """Yield decoded frames without converting non-sampled frames to BGR."""

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate else 30.0
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                timestamp_ms = round(float(frame.pts * frame.time_base) * 1_000)
            else:
                timestamp_ms = round(decoded_index / average_rate * 1_000)
            yield timestamp_ms, decoded_index, frame


def _to_bgr_image(frame: Any) -> np.ndarray:
    """Convert a sampled PyAV frame to BGR while supporting array-based tests."""

    to_ndarray = getattr(frame, "to_ndarray", None)
    if callable(to_ndarray):
        return np.asarray(to_ndarray(format="bgr24"))
    return np.asarray(frame)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a tensor-like Ultralytics output to a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YOLOTrackingService:
    """Run YOLO26 detection and ByteTrack association independently per shot."""

    model_name = "YOLO26"
    tracker_name = "ByteTrack"

    def __init__(
        self,
        *,
        config: TrackingConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or TrackingConfig()
        self.model_path = self._resolve_project_path(self.config.model_path)
        self.tracker_config_path = self._resolve_project_path(
            self.config.tracker_config_path
        )

        if not self.tracker_config_path.is_file():
            raise FileNotFoundError(
                f"ByteTrack configuration does not exist: {self.tracker_config_path}"
            )

        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    "YOLO26 weight does not exist locally: "
                    f"{self.model_path}. Provide TrackingConfig.model_path; "
                    "automatic model download is disabled."
                )
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ImportError(
                    "ultralytics is required for YOLO26 tracking. The dependency "
                    "must be installed from the project requirements."
                ) from error
            model = YOLO(str(self.model_path))

        self.model = model
        model_names = getattr(self.model, "names", None)
        if not isinstance(model_names, (dict, list)):
            raise ValueError("YOLO model must expose its class names through model.names.")
        validate_coco_names(model_names)

        self.model_version = self.model_path.name
        self.tracker_version = version("ultralytics")

    @staticmethod
    def _resolve_project_path(path: Path) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved

    def track_video(
        self,
        video: VideoMetadata,
        shots: Iterable[ShotMetadata],
    ) -> TrackingBatchResult:
        """Track sampled video frames and reset tracker state at every shot."""

        video_path = Path(video.video_path)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        ordered_shots = sorted(shots, key=lambda item: item.start_ms)
        self._validate_shots(video.video_id, ordered_shots)
        if not ordered_shots:
            return TrackingBatchResult([], [])

        observations: list[TrackObservationResult] = []
        accumulators: dict[int, _TrackAccumulator] = {}
        local_track_ids: dict[tuple[str, str, int], int] = {}
        next_local_track_id = 1
        shot_index = 0
        active_shot_id: str | None = None
        sampled_batch: list[tuple[ShotMetadata, int, int, np.ndarray]] = []
        next_sample_ms = ordered_shots[0].start_ms
        sample_interval_ms = 1_000 / self.config.sampling_fps

        def advance_to_shot(timestamp_ms: int) -> ShotMetadata | None:
            nonlocal shot_index, next_sample_ms

            while (
                shot_index < len(ordered_shots)
                and timestamp_ms >= ordered_shots[shot_index].end_ms
            ):
                shot_index += 1
                if shot_index < len(ordered_shots):
                    next_sample_ms = ordered_shots[shot_index].start_ms

            if shot_index >= len(ordered_shots):
                return None

            shot = ordered_shots[shot_index]
            if timestamp_ms < shot.start_ms:
                return None

            return shot

        def flush_batch() -> None:
            nonlocal next_local_track_id

            if not sampled_batch:
                return
            tracked_frames = self._track_batch(
                [sample[3] for sample in sampled_batch]
            )
            if len(tracked_frames) != len(sampled_batch):
                raise RuntimeError(
                    "YOLO tracking returned a different number of frames than "
                    "the submitted batch."
                )

            for sample, tracked_boxes in zip(
                sampled_batch,
                tracked_frames,
                strict=True,
            ):
                shot, timestamp_ms, frame_idx, image = sample
                image_height, image_width = image.shape[:2]
                for internal_track_id, coco_index, confidence, xyxy in tracked_boxes:
                    normalized_box = self._normalize_box(
                        xyxy,
                        width=image_width,
                        height=image_height,
                    )
                    if normalized_box is None:
                        continue

                    canonical_class = get_canonical_class(coco_index)
                    key = (
                        shot.shot_id,
                        canonical_class.class_id,
                        internal_track_id,
                    )
                    if key not in local_track_ids:
                        local_track_ids[key] = next_local_track_id
                        accumulators[next_local_track_id] = _TrackAccumulator(
                            track_id=next_local_track_id,
                            shot_id=shot.shot_id,
                            class_id=canonical_class.class_id,
                            start_ms=timestamp_ms,
                            end_ms=timestamp_ms,
                            start_frame_idx=frame_idx,
                            end_frame_idx=frame_idx,
                        )
                        next_local_track_id += 1

                    local_track_id = local_track_ids[key]
                    accumulators[local_track_id].add(
                        timestamp_ms,
                        frame_idx,
                        confidence,
                    )
                    x_min, y_min, x_max, y_max = normalized_box
                    observations.append(
                        TrackObservationResult(
                            track_id=local_track_id,
                            frame_idx=frame_idx,
                            timestamp_ms=timestamp_ms,
                            confidence=confidence,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                        )
                    )
            sampled_batch.clear()

        for timestamp_ms, frame_idx, decoded_frame in _iter_video_frames(video_path):
            shot = advance_to_shot(timestamp_ms)
            if shot is None:
                if shot_index >= len(ordered_shots):
                    break
                continue
            if timestamp_ms < next_sample_ms:
                continue

            while next_sample_ms <= timestamp_ms:
                next_sample_ms += sample_interval_ms

            if active_shot_id != shot.shot_id:
                flush_batch()
                self._reset_tracker_state()
                active_shot_id = shot.shot_id

            image = _to_bgr_image(decoded_frame)
            sampled_batch.append((shot, timestamp_ms, frame_idx, image))
            if len(sampled_batch) >= self.config.batch_size:
                flush_batch()

        flush_batch()

        tracks = [
            self._to_track_contract(accumulator)
            for accumulator in accumulators.values()
            if accumulator.observation_count > 0
        ]
        retained_track_ids = {track.track_id for track in tracks}
        retained_observations = [
            observation
            for observation in observations
            if observation.track_id in retained_track_ids
        ]
        return TrackingBatchResult(tracks, retained_observations)

    def _track_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[tuple[int, int, float, np.ndarray]]]:
        """Batch YOLO detection while ByteTrack associates frames in order."""

        if not images:
            return []
        kwargs: dict[str, Any] = {
            "source": images,
            "persist": True,
            "tracker": str(self.tracker_config_path),
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "max_det": self.config.max_detections,
            "classes": list(self.config.class_indices),
            "verbose": False,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device

        results = list(self.model.track(**kwargs))
        if len(results) != len(images):
            raise RuntimeError(
                "YOLO tracking must return exactly one result per input image."
            )
        return [self._tracked_boxes_from_result(result) for result in results]

    @staticmethod
    def _tracked_boxes_from_result(
        result: Any,
    ) -> list[tuple[int, int, float, np.ndarray]]:
        """Convert one Ultralytics tracking result to the pipeline contract."""

        boxes = getattr(result, "boxes", None)
        if boxes is None or not bool(getattr(boxes, "is_track", False)):
            return []

        track_ids_value = getattr(boxes, "id", None)
        if track_ids_value is None:
            return []

        track_ids = _as_numpy(track_ids_value).reshape(-1)
        class_indices = _as_numpy(boxes.cls).reshape(-1)
        confidences = _as_numpy(boxes.conf).reshape(-1)
        boxes_xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
        lengths = {
            len(track_ids),
            len(class_indices),
            len(confidences),
            len(boxes_xyxy),
        }
        if len(lengths) != 1:
            raise ValueError("YOLO tracking output arrays have inconsistent lengths.")

        return [
            (
                int(track_id),
                int(class_index),
                float(confidence),
                np.asarray(xyxy, dtype=np.float32),
            )
            for track_id, class_index, confidence, xyxy in zip(
                track_ids,
                class_indices,
                confidences,
                boxes_xyxy,
            )
            if int(track_id) >= 0
        ]

    def _reset_tracker_state(self) -> None:
        """Reset any Ultralytics trackers already attached to this model."""

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        for tracker in trackers or ():
            reset = getattr(tracker, "reset", None)
            if not callable(reset):
                raise RuntimeError("Ultralytics tracker does not expose reset().")
            reset()

    @staticmethod
    def _normalize_box(
        xyxy: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        if width <= 0 or height <= 0:
            raise ValueError("Tracking images must have positive dimensions.")
        x_min = min(max(float(xyxy[0]) / width, 0.0), 1.0)
        y_min = min(max(float(xyxy[1]) / height, 0.0), 1.0)
        x_max = min(max(float(xyxy[2]) / width, 0.0), 1.0)
        y_max = min(max(float(xyxy[3]) / height, 0.0), 1.0)
        if x_min >= x_max or y_min >= y_max:
            return None
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _validate_shots(video_id: str, shots: list[ShotMetadata]) -> None:
        previous_end = -1
        for shot in shots:
            if shot.video_id != video_id:
                raise ValueError(f"Shot {shot.shot_id} does not belong to video {video_id}.")
            if shot.start_ms < 0 or shot.end_ms <= shot.start_ms:
                raise ValueError(f"Invalid boundaries for shot {shot.shot_id}.")
            if shot.start_ms < previous_end:
                raise ValueError("Shots must not overlap.")
            previous_end = shot.end_ms

    def _to_track_contract(self, item: _TrackAccumulator) -> ObjectTrackResult:
        return ObjectTrackResult(
            shot_id=item.shot_id,
            class_id=item.class_id,
            start_ms=item.start_ms,
            end_ms=max(item.end_ms, item.start_ms + 1),
            observation_count=item.observation_count,
            model_name=self.model_name,
            model_version=self.model_version,
            tracker_name=self.tracker_name,
            tracker_version=self.tracker_version,
            sampling_fps=self.config.sampling_fps,
            mapping_version=self.config.mapping_version,
            start_frame_idx=item.start_frame_idx,
            end_frame_idx=item.end_frame_idx,
            avg_confidence=item.confidence_sum / item.observation_count,
            track_id=item.track_id,
        )


# Preserve existing imports while changing the implementation to YOLO26.
ByteTrackService = YOLOTrackingService
Tracker = YOLOTrackingService


__all__ = [
    "ByteTrackService",
    "Tracker",
    "TrackingBatchResult",
    "YOLOTrackingService",
]
