# Object Tracking

Module tracking nhận một video và toàn bộ shot của video đó. Video chỉ được
decode một lần; tracker được reset khi chuyển sang shot mới.

## Hàm sử dụng trong pipeline

```python
from BackEnd.app.tracking import ByteTrackService, TrackingConfig

tracker = ByteTrackService(
    detector=detector,
    config=TrackingConfig(sampling_fps=2.0),
)

result = tracker.track_video(
    video=video_metadata,
    shots=shot_metadata_list,
)
```

Input:

- `video_metadata`: `VideoMetadata`, cần có `video_path`.
- `shot_metadata_list`: `list[ShotMetadata]` của cùng `video_id`.

Output là `TrackingBatchResult` trong RAM:

```python
result.detections     # list[ObjectDetectionResult]
result.tracks         # list[ObjectTrackResult]
result.observations   # list[TrackObservationResult]
```

## Cách hoạt động

```text
VideoMetadata + list[ShotMetadata]
    -> PyAV decode video một lần
    -> sample frame theo sampling_fps
    -> detector trả ObjectDetectionResult
    -> ByteTrack gán tracker ID
    -> track summaries và observations
```

ByteTrack chạy riêng theo từng `class_id` để không nối nhầm class khác nhau.
Khi sang shot mới, tracker state được reset. Các `detection_id` và `track_id`
trong kết quả là ID tạm trong batch; cần remap sang ID database khi persist.
