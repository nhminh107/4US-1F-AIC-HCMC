"""Track objects across every shot while decoding a video only once."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, Iterator

import av
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from BackEnd.app.contracts.pipeline import (
    ObjectDetectionResult,
    ObjectTrackResult,
    ShotMetadata,
    TrackObservationResult,
    VideoMetadata,
)
from BackEnd.app.object_detection.detector import Detector
from BackEnd.app.object_detection.openimages_jsonl import detect_image_array
from BackEnd.app.object_detection.tfhub_openimages_detector import TFHubOpenImagesDetector
from BackEnd.CONFIG import TrackingConfig


@dataclass(frozen=True, slots=True)
class TrackingBatchResult:
    """In-memory contracts whose integer IDs are local to this batch."""

    detections: list[ObjectDetectionResult]
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


def _iter_video_frames(video_path: Path) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(timestamp_ms, frame_idx, BGR image)`` in decode order."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate else 30.0
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                timestamp_ms = round(float(frame.pts * frame.time_base) * 1_000)
            else:
                timestamp_ms = round(decoded_index / average_rate * 1_000)
            yield timestamp_ms, decoded_index, frame.to_ndarray(format="bgr24")


class ByteTrackService:
    """Decode one video once and reset class-aware ByteTrack state per shot."""

    def __init__(
        self,
        *,
        detector: Detector | None = None,
        config: TrackingConfig | None = None,
    ) -> None:
        self.detector = detector or TFHubOpenImagesDetector(confidence_threshold=0.10)
        self.config = config or TrackingConfig()
        self.tracker_version = version("trackers")

    def _new_tracker(self) -> ByteTrackTracker:
        return ByteTrackTracker(
            lost_track_buffer=self.config.lost_track_buffer,
            frame_rate=self.config.sampling_fps,
            track_activation_threshold=self.config.track_activation_threshold,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=self.config.minimum_iou_threshold,
            high_conf_det_threshold=self.config.high_confidence_threshold,
        )

    def track_video(
        self,
        video: VideoMetadata,
        shots: Iterable[ShotMetadata],
    ) -> TrackingBatchResult:
        video_path = Path(video.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        ordered_shots = sorted(shots, key=lambda item: item.start_ms)
        self._validate_shots(video.video_id, ordered_shots)
        if not ordered_shots:
            return TrackingBatchResult([], [], [])

        detections: list[ObjectDetectionResult] = []
        observations: list[TrackObservationResult] = []
        accumulators: dict[int, _TrackAccumulator] = {}
        tracker_ids: dict[tuple[str, str, int], int] = {}
        next_track_id = 1
        shot_index = 0
        active_shot_id: str | None = None
        class_trackers: dict[str, ByteTrackTracker] = {}
        next_sample_ms = ordered_shots[0].start_ms
        sample_interval_ms = 1_000 / self.config.sampling_fps

        for timestamp_ms, frame_idx, image in _iter_video_frames(video_path):
            while (
                shot_index < len(ordered_shots)
                and timestamp_ms >= ordered_shots[shot_index].end_ms
            ):
                shot_index += 1
                active_shot_id = None
                class_trackers = {}
                if shot_index < len(ordered_shots):
                    next_sample_ms = ordered_shots[shot_index].start_ms

            if shot_index >= len(ordered_shots):
                break

            shot = ordered_shots[shot_index]
            if timestamp_ms < shot.start_ms or timestamp_ms < next_sample_ms:
                continue

            if active_shot_id != shot.shot_id:
                active_shot_id = shot.shot_id
                class_trackers = {}

            while next_sample_ms <= timestamp_ms:
                next_sample_ms += sample_interval_ms

            frame_id = f"{video.video_id}_tracking_{frame_idx:08d}"
            frame_detections = detect_image_array(
                image,
                frame_id=frame_id,
                detector=self.detector,
            )
            frame_detections = [
                replace(item, detection_id=len(detections) + offset + 1)
                for offset, item in enumerate(frame_detections)
            ]
            detection_start = len(detections)
            detections.extend(frame_detections)

            grouped_indices: dict[str, list[int]] = {}
            for local_index, detection in enumerate(frame_detections):
                grouped_indices.setdefault(detection.class_id, []).append(local_index)

            height, width = image.shape[:2]
            for class_id, local_indices in grouped_indices.items():
                tracker = class_trackers.get(class_id)
                if tracker is None:
                    tracker = self._new_tracker()
                    class_trackers[class_id] = tracker
                tracker_input = self._to_tracker_input(
                    frame_detections,
                    local_indices,
                    width=width,
                    height=height,
                )
                tracked = tracker.update(
                    tracker_input,
                    timestamp=timestamp_ms / 1_000,
                )

                for row_index, internal_track_id in enumerate(tracked.tracker_id):
                    if internal_track_id < 0:
                        continue
                    local_index = int(tracked.data["local_index"][row_index])
                    detection = frame_detections[local_index]
                    key = (shot.shot_id, class_id, int(internal_track_id))
                    if key not in tracker_ids:
                        tracker_ids[key] = next_track_id
                        accumulators[next_track_id] = _TrackAccumulator(
                            track_id=next_track_id,
                            shot_id=shot.shot_id,
                            class_id=class_id,
                            start_ms=timestamp_ms,
                            end_ms=timestamp_ms,
                            start_frame_idx=frame_idx,
                            end_frame_idx=frame_idx,
                        )
                        next_track_id += 1

                    track_id = tracker_ids[key]
                    accumulators[track_id].add(
                        timestamp_ms,
                        frame_idx,
                        detection.confidence,
                    )
                    observations.append(
                        TrackObservationResult(
                            track_id=track_id,
                            detection_id=detection_start + local_index + 1,
                        )
                    )

        tracks = [self._to_track_contract(item) for item in accumulators.values()]
        return TrackingBatchResult(detections, tracks, observations)

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

    @staticmethod
    def _to_tracker_input(
        detections: list[ObjectDetectionResult],
        indices: list[int],
        *,
        width: int,
        height: int,
    ) -> sv.Detections:
        xyxy = np.asarray(
            [
                [
                    detections[index].x_min * width,
                    detections[index].y_min * height,
                    detections[index].x_max * width,
                    detections[index].y_max * height,
                ]
                for index in indices
            ],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [detections[index].confidence for index in indices],
            dtype=np.float32,
        )
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            data={"local_index": np.asarray(indices, dtype=np.int64)},
        )

    def _to_track_contract(self, item: _TrackAccumulator) -> ObjectTrackResult:
        return ObjectTrackResult(
            shot_id=item.shot_id,
            class_id=item.class_id,
            start_ms=item.start_ms,
            end_ms=max(item.end_ms, item.start_ms + 1),
            observation_count=item.observation_count,
            start_frame_idx=item.start_frame_idx,
            end_frame_idx=item.end_frame_idx,
            avg_confidence=item.confidence_sum / item.observation_count,
            tracker_name="ByteTrack",
            tracker_version=self.tracker_version,
            track_id=item.track_id,
        )


# Backward-compatible name for existing imports.
Tracker = ByteTrackService
