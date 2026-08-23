# Kế hoạch tái cấu trúc Object Tracking

> Trạng thái: implementation đã hoàn thành ngày 2026-08-11; unit/focused tests
> đã đạt. GPU smoke test đang chờ weight/video cục bộ. Migration chỉ được bàn
> giao để người dùng tự áp dụng, không được tự chạy trên database.

## 1. Mục tiêu

Giảm đáng kể tài nguyên tính toán của Tracking bằng cách:

- Không chạy Faster R-CNN Inception-ResNet-v2 bên trong Tracking.
- Dùng YOLO26 qua `ultralytics.YOLO.track()` với tracker ByteTrack.
- Chỉ detect và track 80 lớp lớn của COCO.
- Map thủ công 80 lớp COCO sang `class_id` và `class_name` tương ứng trong
  Open Images v4.
- Coi Object Detection trên keyframe và Tracking trên video/shot là hai nguồn
  kết quả độc lập.
- Không sửa, import hoặc tái sử dụng module Object Detection trong Tracking.
- Không tạo `ObjectDetection` từ các frame tạm dùng riêng cho Tracking.
- Không liên kết một track với `ObjectDetection` thông qua `detection_id`.

## 2. Hiện trạng trước thay đổi đã kiểm chứng

Implementation hiện tại có các đặc điểm sau:

1. `ByteTrackService` mặc định khởi tạo
   `TFHubOpenImagesDetector`, tức Faster R-CNN Inception-ResNet-v2 trên Open
   Images.
2. Video được decode một lần và lấy mẫu mặc định ở 2 FPS.
3. Mỗi frame được chọn vẫn phải chạy Faster R-CNN trước khi kết quả được đưa
   vào ByteTrack.
4. Tracker được reset khi chuyển sang shot mới.
5. Tracking hiện tạo ba loại contract:
   `ObjectDetectionResult`, `ObjectTrackResult` và `TrackObservationResult`.
6. Bảng `TrackObservation` nối `ObjectTrack.track_id` với
   `ObjectDetection.detection_id`.
7. `ObjectTrack` chỉ là bản tóm tắt một track theo shot; bảng này không phải
   nguyên nhân gây tải GPU đáng kể.
8. `ultralytics`, `trackers` và `supervision` đều đang được khai báo/cài đặt.
   Không có YOLO weight (`.pt`, `.onnx` hoặc `.engine`) trong repository tại
   thời điểm lập kế hoạch.
9. `BackEnd/app/pipeline/tracking.py` đang có thay đổi chưa commit và hiện gọi
   `result.classes` cùng `db.persist_tracking_results()`, trong khi hai API này
   không tồn tại trong contract/database hiện tại. Khi triển khai phải bảo toàn
   thay đổi của người dùng và thay phần đang dở bằng pipeline mới nhất quán.

Ước lượng trước đây cho thấy Tracking phải xử lý khoảng 2,448 triệu frame tạm
cho 340 giờ video ở 2 FPS. Tải chính nằm ở detector chạy trên từng frame, không
nằm ở phép liên kết database.

## 3. Kiến trúc đích

```text
Object Detection pipeline
    -> Giữ nguyên hoàn toàn, nằm ngoài phạm vi thay đổi

Tracking pipeline
Video -> decode một lần -> sample theo tracking FPS
    -> YOLO.track(..., tracker="bytetrack.yaml")
    -> map COCO class sang Open Images class
    -> tổng hợp theo shot và track ID
    -> ObjectTrack
```

Hai pipeline chỉ dùng chung ở tầng dữ liệu:

- `Shot`/thông tin thời gian để xác định phạm vi track.
- `ClassID` chuẩn để hỗ trợ truy vấn chung theo semantic class.

Hai pipeline không dùng chung detection record, bounding box record hoặc
confidence. Confidence của Faster R-CNN và YOLO không được coi là cùng một
thang đo để so sánh hoặc gộp trực tiếp.

## 4. Quyết định database đề xuất

### 4.1 Giữ `ObjectTrack`

Giữ bảng `ObjectTrack` vì đây là đầu ra chính của Tracking, bao gồm:

- Shot chứa object.
- Class của object.
- Khoảng thời gian và khoảng frame object xuất hiện.
- Số lần object được quan sát.
- Confidence trung bình.
- Thông tin detector/tracker cần cho truy vết kết quả.

Xóa `ObjectTrack` sẽ làm mất nơi lưu track summary và không giúp giảm đáng kể
chi phí inference.

### 4.2 Đổi `TrackObservation` thành observation độc lập

Giữ bảng và contract `TrackObservation`, nhưng thay cấu trúc hiện tại
`(track_id, detection_id)` bằng observation do YOLO trực tiếp tạo ra:

- `track_id`
- `frame_idx`
- `timestamp_ms`
- normalized bbox `x_min`, `x_max`, `y_min`, `y_max`
- `confidence`

Khóa chính mới là `(track_id, frame_idx)`. Xóa `detection_id`, foreign key tới
`ObjectDetection` và relationship tương ứng. `TrackObservation.track_id` vẫn là
foreign key tới `ObjectTrack.track_id`.

Migration thay cấu trúc bảng cũ là thao tác có rủi ro mất liên kết legacy. File
migration phải sao lưu dữ liệu cũ trước khi tạo cấu trúc mới và chỉ được bàn
giao; không chạy migration trên database thật.

### 4.3 Metadata của Tracking

Đề xuất bổ sung vào `ObjectTrack`:

- `model_name`: tên detector YOLO.
- `model_version`: weight/version cụ thể.
- `sampling_fps`: FPS thực tế đưa vào tracker.
- `mapping_version`: phiên bản mapping COCO -> Open Images.

Giữ `tracker_name` và `tracker_version` hiện có. Các trường trên tránh việc có
track nhưng không thể xác định model, sampling hoặc mapping đã sinh ra nó.

`TrackObservation` mới chính là dữ liệu trajectory để vẽ đường đi hoặc lấy bbox
tại từng thời điểm; không cần thêm bảng `TrackPoint` riêng.

## 5. Quy ước class

### 5.1 Vocabulary chuẩn

Sử dụng Open Images v4 làm canonical vocabulary vì:

- Faster R-CNN hiện trả trực tiếp Open Images MID.
- Bảng `ClassID` hiện được nạp từ `open_images_v4_classes.csv`.
- `ObjectDetection.class_id` đã sử dụng Open Images MID.

YOLO không được persist ID dạng `c000`, `c001`, ... cho Tracking. Thay vào đó,
COCO class index/name được map sang Open Images MID/name trước khi tạo
`ObjectTrackResult`.

Ví dụ:

```text
YOLO: index=0, name="person"
    -> Open Images: class_id="/m/01g317", class_name="Person"
```

### 5.2 Kết quả đối chiếu vocabulary

Đã đối chiếu 80 class COCO trong code với 601 class trong
`open_images_v4_classes.csv`:

- 66/80 class khớp trực tiếp theo tên sau khi chuẩn hóa hoa/thường.
- 14/80 class cần map alias thủ công.

Các alias dự kiến:

| COCO class | Open Images class | Open Images class_id | Ghi chú |
|---|---|---|---|
| cow | Cattle | `/m/01xq0k1` | Open Images rộng hơn COCO |
| frisbee | Flying disc | `/m/02wmf` | Tương đương thực dụng |
| skis | Ski | `/m/071p9` | Khác số ít/số nhiều |
| sports ball | Ball | `/m/018xm` | Open Images rộng hơn COCO |
| cup | Coffee cup | `/m/02p5f1q` | Không hoàn toàn tương đương; cần duyệt |
| donut | Doughnut | `/m/0jy4k` | Khác cách viết |
| potted plant | Houseplant | `/m/03fp41` | Gần tương đương; cần duyệt |
| dining table | Kitchen & dining room table | `/m/0h8n5zk` | Tương đương thực dụng |
| tv | Television | `/m/07c52` | Tương đương |
| remote | Remote control | `/m/0qjjc` | Tương đương |
| keyboard | Computer keyboard | `/m/01m2v` | Loại trừ musical keyboard |
| cell phone | Mobile phone | `/m/050k8` | Tương đương |
| microwave | Microwave oven | `/m/0fx9l` | Tương đương |
| hair drier | Hair dryer | `/m/03wvsk` | Sửa khác biệt chính tả |

Bốn mapping cần được chấp nhận như quy ước semantic của dự án:

- `cow -> Cattle`
- `sports ball -> Ball`
- `cup -> Coffee cup`
- `potted plant -> Houseplant`

### 5.3 Yêu cầu đối với mapping

Mapping phải:

- Chứa đúng 80 COCO indices từ 0 đến 79.
- Khóa theo cả COCO index và expected class name.
- Không có index trùng hoặc Open Images target ngoài CSV.
- Fail fast nếu class order/name của YOLO weight không khớp mapping.
- Có hằng `mapping_version`, ví dụ `coco80-openimages-v1`.
- Có unit test kiểm tra đầy đủ 80 entries và tính duy nhất.

### 5.4 Lọc Faster R-CNN

Module Object Detection và Faster R-CNN được giữ nguyên hoàn toàn. Giới hạn 80
lớp chỉ áp dụng cho YOLO26 Tracking. Mapping được đặt trong module Tracking và
không import code, mapper hoặc CSV từ module Object Detection.

## 6. Hành vi Tracking mới

### 6.1 Model và tracker

- Dùng một YOLO model được load một lần cho mỗi worker/process.
- Model path mặc định là `yolo26n.pt`, có thể cấu hình và phải trỏ tới weight cục
  bộ; không tải weight ngầm trong production.
- Gọi `model.track()` với `tracker="bytetrack.yaml"` một cách tường minh, không
  phụ thuộc tracker mặc định của Ultralytics.
- Dùng `persist=True` chỉ cho các frame liên tiếp thuộc cùng một shot.
- Reset tracker bắt buộc khi đổi shot hoặc video.
- Đọc track ID từ `result.boxes.id` chỉ khi `result.boxes.is_track` là true.
- Bbox từ Tracking không được chuyển thành `ObjectDetectionResult`.

Việc reset state của Ultralytics sẽ được đóng gói trong adapter riêng và có unit
test, tránh rải truy cập internal predictor state khắp pipeline.

### 6.2 Sampling

- Giữ default 2 FPS ở lần triển khai đầu để bảo toàn mục tiêu tài nguyên.
- Decode video đúng một lần.
- Chỉ gọi YOLO cho frame đạt mốc sampling.
- `track_buffer` được hiểu theo số frame đã sample, không phải FPS gốc của video.
- Benchmark sau triển khai phải so sánh ít nhất 2 FPS và 5 FPS trên video đại
  diện trước khi chạy toàn bộ 340 giờ.

Không gọi thẳng `model.track(video_path)` nếu cách gọi đó làm inference trên toàn
bộ FPS gốc. Pipeline phải kiểm soát rõ frame nào được đưa vào model.

### 6.3 Phạm vi track ID

YOLO track ID chỉ có ý nghĩa tạm thời trong một shot. Trong RAM, khóa accumulator
là `(shot_id, yolo_track_id)`. Khi persist, PostgreSQL sinh `ObjectTrack.track_id`
toàn cục như hiện tại.

### 6.4 Kết quả trả về

Tracking mới trả `TrackingBatchResult` gồm:

- `tracks: list[ObjectTrackResult]`
- `observations: list[TrackObservationResult]`

Nó không trả:

- `ObjectDetectionResult`
- Detection tạm của Tracking

Pipeline persistence chỉ thêm `ObjectTrack`.

## 7. Cấu hình đề xuất

Mở rộng `TrackingConfig` với các trường tối thiểu:

```text
model_path
device
sampling_fps
confidence_threshold
iou_threshold
max_detections
tracker_config_path
mapping_version
```

Các threshold cụ thể của ByteTrack nằm trong YAML tracker được version-control.
Không dùng giá trị mặc định ẩn của thư viện nếu giá trị đó ảnh hưởng kết quả.

Model mặc định được phê duyệt là YOLO26 nano (`yolo26n.pt`) để ưu tiên tài
nguyên. Model path vẫn configurable để có thể benchmark weight YOLO26 khác mà
không sửa source code.

## 8. Kế hoạch thực hiện chi tiết

### Giai đoạn 1 - Khóa mapping 80 lớp

1. Tạo module mapping COCO -> Open Images dùng chung.
2. Khai báo đủ 80 entry với source index/name và target MID/name.
3. Load CSV để validate target ID/name.
4. Validate class names của YOLO model khi khởi tạo Tracking.
5. Thêm unit test cho coverage, uniqueness, alias và mismatch.

Kết quả: một mapping xác định, có version và không thể âm thầm map sai class.

### Giai đoạn 2 - Khóa phạm vi Object Detection

1. Không sửa file nào trong `BackEnd/app/object_detection/`.
2. Không import `Detector`, `ClassMapper`, parser, post-processing hoặc CSV của
   Object Detection vào Tracking.
3. Dùng test/phép kiểm tra phạm vi để đảm bảo thay đổi không chạm module này.

Kết quả: behavior và contract của Object Detection không thay đổi.

### Giai đoạn 3 - Viết YOLO Tracking service

1. Thay implementation Faster R-CNN + external ByteTrack trong module Tracking
   bằng adapter dùng `YOLO.track()`.
2. Giữ single-pass video decoding và sampling có timestamp/frame index.
3. Reset tracker theo ranh giới shot/video.
4. Map COCO class sang Open Images class trước khi accumulate.
5. Tổng hợp start/end, frame range, observation count và average confidence.
6. Không load persisted keyframe và không sinh detection contract.
7. Sinh `TrackObservationResult` chứa timestamp, frame index, normalized bbox
   và confidence của YOLO.
8. Xử lý rõ frame không có box, box chưa có track ID và lỗi class mapping.

Kết quả: Tracking độc lập hoàn toàn với Object Detection ở runtime.

### Giai đoạn 4 - Cập nhật contracts và pipeline persistence

1. Đổi `TrackObservationResult` sang schema observation độc lập.
2. Thu gọn `TrackingBatchResult` còn tracks và observations.
3. Bổ sung provenance fields đã duyệt vào `ObjectTrackResult`.
4. Viết lại `BackEnd/app/pipeline/tracking.py` để persist `ObjectTrack` và
   `TrackObservation` trong cùng transaction.
5. Loại bỏ logic chọn persisted detection; chỉ remap local track ID sang database
   track ID.
6. Không ghi đè mù quáng thay đổi chưa commit hiện có trong file pipeline.

Kết quả: service, contract và orchestration thống nhất với kiến trúc mới.

### Giai đoạn 5 - Cập nhật database/schema

1. Đổi model/contract `TrackObservation` sang các field YOLO observation.
2. Đổi schema tạo database mới và bỏ foreign key tới `ObjectDetection`.
3. Bổ sung metadata Tracking đã duyệt vào `ObjectTrack`.
4. Cập nhật database adapter/method theo observation schema mới.
5. Tạo migration SQL có transaction và backup bảng observation cũ cho database
   hiện hữu.
6. Chỉ bàn giao migration; không thực thi trên database thật.

Kết quả: `ObjectDetection` và `ObjectTrack` không còn foreign key hoặc bảng nối
trực tiếp với nhau.

### Giai đoạn 6 - Dọn dependency và tài liệu

1. Nếu không còn import nào, bỏ `trackers` và `supervision` khỏi requirements.
2. Không cài, nâng cấp hoặc hạ phiên bản package trong quá trình này.
3. Cập nhật README Tracking và mô tả pipeline; không sửa tài liệu/module Object
   Detection.
4. Cập nhật project instructions nói Tracking trả detection observations vì mô
   tả đó sẽ không còn đúng.
5. Ghi rõ yêu cầu chuẩn bị YOLO weight cục bộ trước khi chạy cloud.

Kết quả: dependency và tài liệu phản ánh đúng implementation.

### Giai đoạn 7 - Kiểm thử

Unit tests tối thiểu:

- Mapping đủ 80 lớp, ID/name target đúng snapshot đã duyệt.
- YOLO class order sai thì fail fast.
- Sampling chỉ inference đúng frame cần thiết.
- Model được load một lần.
- Tracker reset đúng tại shot/video boundary.
- Track ID không rò giữa hai shot.
- Aggregate timestamp/frame/confidence chính xác.
- Frame không có track ID không tạo track lỗi.
- Tracking không tạo `ObjectDetectionResult`.
- Mỗi track observation chứa đúng timestamp, frame index, bbox và confidence.
- Pipeline persist `ObjectTrack` và `TrackObservation` trong cùng transaction.
- Database model/schema giữ `TrackObservation` nhưng không còn `detection_id`.
- Không có thay đổi nào trong module Object Detection.

Các lệnh kiểm tra sau khi code:

```bash
source ~/miniconda3/etc/profile.d/conda.sh \
  && conda activate DL_Env \
  && python -m compileall BackEnd/app BackEnd/tests

source ~/miniconda3/etc/profile.d/conda.sh \
  && conda activate DL_Env \
  && pytest -q BackEnd/tests/tracking \
       BackEnd/tests/pipeline/test_tracking_pipeline.py \
       BackEnd/tests/object_detection \
       BackEnd/tests/database/test_adapter.py
```

Sau focused tests, chạy full test suite nếu thời gian cho phép.

Real-model smoke test chỉ chạy khi có YOLO weight cục bộ. Repository hiện không
có weight; không tự động download model nếu chưa được cho phép.

### Giai đoạn 8 - Benchmark trước khi chạy toàn bộ dữ liệu

Benchmark trên một tập video đại diện, có shot ngắn/dài, camera chuyển động,
đông người và object nhỏ:

- Wall-clock time và số frame/giây.
- Peak GPU memory.
- Số track theo class.
- Tỷ lệ track rất ngắn.
- ID switch qua che khuất/chuyển động.
- So sánh 2 FPS và 5 FPS.
- So sánh ít nhất hai YOLO weight nếu weight có sẵn.

Chỉ chạy toàn bộ 340 giờ sau khi benchmark xác nhận tốc độ và chất lượng chấp
nhận được.

## 9. Danh sách file dự kiến thay đổi

Phạm vi chính dự kiến gồm:

- `BackEnd/CONFIG.py`
- Module mapping COCO -> Open Images mới
- `BackEnd/app/object_detection/tfhub_openimages_detector.py`
- `BackEnd/app/tracking/tracking.py`
- `BackEnd/app/tracking/__init__.py`
- `BackEnd/app/contracts/pipeline.py`
- `BackEnd/app/pipeline/tracking.py`
- `BackEnd/app/database/models.py`
- `BackEnd/app/database/adapter.py`
- `BackEnd/app/database/postgre_db.py`
- `BackEnd/app/database/postgre_script.sql`
- Migration SQL mới trong `BackEnd/app/database/migrations/`
- `pyproject.toml` và `uv.lock`
- README/instructions liên quan
- Các test Tracking, pipeline, database và Object Detection liên quan

Không thay đổi dataset, model weight, FAISS index, database thật hoặc file chứa
credentials.

## 10. Tiêu chí hoàn thành

Thay đổi chỉ được coi là hoàn thành khi:

1. Tracking không import hoặc khởi tạo Faster R-CNN.
2. Tracking dùng `YOLO.track()` và ByteTrack được chọn tường minh.
3. Chỉ các frame theo sampling FPS được inference.
4. Tracker reset đúng theo shot/video.
5. Đủ 80 COCO class được map sang Open Images canonical class.
6. Module Object Detection không thay đổi.
7. Tracking không tạo hoặc persist `ObjectDetection`.
8. `TrackObservation` lưu trực tiếp YOLO observation và không phụ thuộc
   `ObjectDetection`.
9. `ObjectTrack` còn được persist với provenance đầy đủ.
10. Focused tests và syntax checks chạy thành công.
11. Mọi test chưa chạy hoặc real-model benchmark chưa thực hiện được phải được
    báo cáo rõ.

## 11. Các điểm cần phê duyệt trước khi code

1. Giữ `ObjectTrack` và `TrackObservation`; đổi observation sang timestamp,
   frame index, bbox, confidence và xóa liên kết với `ObjectDetection`.
2. Dùng Open Images MID/name làm canonical class cho Tracking mà không sửa hoặc
   tái sử dụng module Object Detection.
3. Chấp nhận bốn mapping gần tương đương:
   `cow -> Cattle`, `sports ball -> Ball`, `cup -> Coffee cup`,
   `potted plant -> Houseplant`.
4. Giữ 2 FPS làm default ban đầu và benchmark thêm 5 FPS.
5. Dùng YOLO26 nano (`yolo26n.pt`) làm mặc định, nhưng để model path configurable.
6. Bổ sung provenance fields vào `ObjectTrack`.
7. Giữ `TrackObservation` nhưng đổi thành YOLO observation độc lập; chỉ tạo và
   bàn giao migration, không chạy trên database thật.
