# Kế hoạch cloud computing cho pipeline hiện tại

## Thay đổi ảnh hưởng đến chi phí tính toán

Pipeline đã giảm đáng kể workload GPU so với thiết kế ban đầu:

- ObjectDetection cho keyframe BTC được nạp trực tiếp từ
  `data/objects-aic25-b1-jsonl`; không chạy lại Faster R-CNN trên toàn bộ
  video.
- Tracking độc lập với ObjectDetection. YOLO26 + ByteTrack decode video một
  lần, lấy mẫu mặc định 2 FPS và chỉ infer 22 COCO class có giá trị temporal.
- OCR chạy theo batch với một `OCRService` dùng chung.
- Frame, clip và shot embedding là ba stage riêng nhưng dùng chung một
  `EmbeddingPipeline` và một FAISS manager. Clip embedding có thể được tái sử
  dụng khi aggregate shot embedding.
- Clip mặc định chỉ là metadata (`materialize_files=False`), không encode lại
  MP4.

Do đó, ước lượng cũ cho Tracking dựa trên Faster R-CNN từng frame không còn áp
dụng. Giá thuê GPU/khuyến mãi phải được xác nhận lại tại thời điểm chạy; tài
liệu này chỉ ước lượng workload và thời gian GPU.

## Quy mô workload

- Khoảng 340 giờ video.
- Tracking tại 2 FPS tương ứng khoảng `340 × 3,600 × 2 = 2,448,000` frame
  lấy mẫu.
- Object JSONL có 177,321 file keyframe organizer; việc import là I/O +
  PostgreSQL, không phải inference GPU.
- Tổng số keyframe sau khi extract phụ thuộc kết quả Shot/Keyframe; không nên
  cố định giả định 300,000 trước khi benchmark.

## Phân bổ công việc đề xuất

| Stage | Tài nguyên phù hợp | Ước lượng hiện tại | Ghi chú |
| --- | --- | --- | --- |
| Ingest `ClassID`, Frame, ObjectDetection JSONL | CPU + PostgreSQL | Không dùng GPU; đo theo I/O/DB | Object JSONL insert theo batch 1,000 record và không query Frame từng record. Nên chạy gần PostgreSQL. |
| Shot, additional keyframe, logical clip | RTX 3090 Ti hoặc tương đương | 20–50 giờ GPU | Phần chính là decode video + TransNetV2 + export keyframe. Logical clip gần như không đáng kể. Xuất MP4 clip sẽ tăng mạnh thời gian và dung lượng. |
| OCR | RTX 3090 Ti hoặc tương đương | 8–25 giờ GPU | Phụ thuộc mật độ chữ và throughput OCR end-to-end. Model phải được chuẩn bị sẵn trên máy cloud. |
| ASR | H100/A100 hoặc GPU có VRAM lớn | 15–35 giờ GPU | Bao gồm extract/normalize audio, VAD và ChunkFormer. Cần benchmark I/O cùng GPU. |
| Tracking | RTX 3090 Ti hoặc tương đương | Budget ban đầu 10–35 giờ GPU | YOLO26n + ByteTrack ở 2 FPS, chỉ 22 class. Đây là estimate end-to-end cần xác thực, không phải FPS inference thuần. |
| Frame/clip/shot embedding | RTX 3090 Ti hoặc tương đương | 8–24 giờ GPU | Frame và clip là workload chính. Shot là pooling từ clip embedding, thường rất nhỏ sau khi clip đã có. |

### Cách suy ra estimate Tracking

Tracking phải đo toàn bộ chuỗi `PyAV decode -> lấy mẫu -> YOLO26.track ->
ByteTrack -> ghi PostgreSQL`, không chỉ đo forward pass. Với 2,448,000 frame
lấy mẫu:

| Throughput end-to-end | Thời gian cho Tracking |
| --- | --- |
| 20 frame/giây | ~34 giờ |
| 35 frame/giây | ~19 giờ |
| 60 frame/giây | ~11 giờ |

Whitelist 22 class giảm số candidate box, association và số row cần ghi DB;
không thay đổi số frame phải decode. Vì vậy bottleneck thực tế có thể chuyển từ
GPU sang decode/video I/O hoặc PostgreSQL.

## Thứ tự chạy khuyến nghị

1. Nạp `ClassID`, video metadata và official Frame.
2. Nạp `ObjectDetection` từ JSONL. Đây là nguồn độc lập với Tracking.
3. Chạy Shot -> additional keyframe -> logical clip.
4. Chạy OCR trên frame đã có.
5. Chạy Tracking trên video + shot. Không cần ObjectDetection runtime.
6. Chạy frame embedding -> clip embedding -> shot embedding.
7. Chạy ASR độc lập; có thể dùng GPU khác song song với visual stages.

Khi chỉ có một GPU 24 GB, không chạy đồng thời OCR, Tracking và Embedding.
Các model nặng nên được load một lần cho từng stage và giải phóng trước stage
kế tiếp để tránh tranh chấp VRAM.

## Benchmark trước khi thuê dài hạn

Chạy mẫu đại diện 5–10 video hoặc 2–5 giờ video, bao gồm cả video ngắn/dài,
nhiều/ít shot. Ghi tối thiểu:

- số video, shot, frame/keyframe/clip và record DB đã tạo;
- throughput end-to-end cho OCR, Tracking và clip embedding;
- GPU utilization, VRAM, CPU utilization, tốc độ đọc video và batch size;
- thời gian PostgreSQL insert cho ObjectDetection JSONL và tracking
  observations;
- số record bị thiếu media, skip hoặc lỗi.

Chỉ extrapolate chi phí sau benchmark này. Nếu Tracking thấp hơn 20 sampled
frame/giây, ưu tiên kiểm tra decode/I/O và PostgreSQL trước khi đổi sang model
lớn hơn hoặc tăng GPU.
