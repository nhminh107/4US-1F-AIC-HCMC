# Object Tracking

Tracking chạy YOLO26 + ByteTrack độc lập với module Object Detection. Video
được decode đúng một lần, lấy mẫu theo `sampling_fps`, và tracker được reset ở
mỗi ranh giới shot.

## Cấu hình

Weight mặc định được tìm tại:

```text
data/models/yolo26n.pt
```

Pipeline không tự tải weight. Có thể truyền đường dẫn weight YOLO26 khác:

```python
from pathlib import Path

from BackEnd.app.tracking import TrackingConfig, YOLOTrackingService

tracker = YOLOTrackingService(
    config=TrackingConfig(
        model_path=Path("/models/yolo26s.pt"),
        sampling_fps=2.0,
        device="cuda:0",
    )
)
```

Model phải có đúng COCO 80 class theo thứ tự chuẩn. Tracking map từng class sang
Open Images MID bằng mapping version `coco80-openimages-v1`.

Mặc định YOLO chỉ tracking 22 class có giá trị temporal cao: người, phương
tiện, một số động vật, vật mang theo và dụng cụ thể thao. Có thể thay đổi bằng
`TrackingConfig(class_indices=(...))`. Object Detection vẫn hoạt động độc lập
và không bị giới hạn bởi cấu hình này.

## Input và output

```python
result = tracker.track_video(
    video=video_metadata,
    shots=shot_metadata_list,
)

result.tracks        # list[ObjectTrackResult]
result.observations  # list[TrackObservationResult]
```

`TrackObservationResult` lưu trực tiếp timestamp, frame index, normalized bbox
và confidence từ YOLO. Nó không có `detection_id` và không tham chiếu
`ObjectDetection`.

```text
VideoMetadata + list[ShotMetadata]
    -> PyAV decode một lần
    -> sample frame theo sampling_fps
    -> YOLO26 detect + ByteTrack associate
    -> map COCO class sang Open Images class
    -> ObjectTrack summaries + TrackObservation trajectory
```

## Lưu ý vận hành

- `persist=True` chỉ được dùng giữa các frame thuộc cùng shot.
- Tracker state phải reset khi đổi shot hoặc video.
- `track_buffer` trong `bytetrack.yaml` được tính theo số frame đã sample.
- Benchmark 2 FPS và 5 FPS trước khi xử lý toàn bộ dataset.
- Không so sánh trực tiếp confidence YOLO với confidence Faster R-CNN.
