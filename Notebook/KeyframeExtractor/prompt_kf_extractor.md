# YÊU CẦU TRIỂN KHAI: KAGGLE NOTEBOOK — ADDITIONAL KEYFRAME EXTRACTOR

Hãy tạo một Kaggle Notebook độc lập để trích xuất **additional keyframe** từ video BTC. Notebook không kết nối trực tiếp tới PostgreSQL: đầu vào là các bảng đã export, đầu ra là file SQL để chạy ở môi trường có database. JPEG trích xuất phải upload thẳng lên Cloudflare R2 theo cấu hình được cung cấp.

Mục tiêu là tái hiện đúng contract của module `BackEnd/app/keyframe_extractor`, nhưng chạy hiệu quả trên một GPU Kaggle. Không làm caption hay ASR; không tạo, sửa hoặc upload keyframe BTC.

## 1. Deliverable

Tạo notebook `Notebook/KeyframeExtractor/keyframe_extractor_kaggle.ipynb` cùng các file nhỏ cần thiết trong cùng thư mục. Notebook phải chạy lần lượt từ đầu đến cuối và có các phần rõ ràng: cấu hình, đọc/validate input, download, select, extract JPEG, upload R2, tạo SQL, checkpoint/resume, report, mock test.

Không cần production CLI hoàn chỉnh. Tuy nhiên các cell phải idempotent và có thể resume sau khi Kaggle timeout.

## 2. Input contract

Người dùng sẽ upload/attach các file sau vào Kaggle Dataset; đường dẫn phải cấu hình được ở đầu notebook.

1. `video.txt`: gồm ít nhất `video_id,video_url`. `video_url` là URL tải MP4 (thường là R2/public URL)., hỗ trợ một dòng `video_id,url`.
2. `shot.csv` (export bảng `shot`), đúng các cột:
   `shot_id, video_id, shot_index, start_ms, end_ms, start_frame_idx, end_frame_idx`.
3. `keyframe.csv` (export bảng `frame`), ít nhất các cột:
   `frame_id, video_id, shot_id, timestamp_ms, fps, frame_idx, source, n, pts_time, frame_path, width, height`.

`keyframe.csv` ban đầu chứa keyframe BTC với `source='official'`; có thể đã chứa frame `source='extracted'` của các lần chạy trước. Hãy coi **mọi** `frame_idx` đã có của cùng video là existing frame để chống trùng. Tuyệt đối không giả định `shot_id` của keyframe BTC khác `NULL` — contract đúng là `NULL`.

Validate trước khi chạy:

- video ID trong `shot.csv` phải có URL;
- `shot_index` không trùng trong một video và xử lý theo thứ tự tăng dần;
- frame bounds không âm, `end_frame_idx >= start_frame_idx`, `fps > 0`;
- không có `frame_id` trùng;
- chỉ xử lý video có shot hợp lệ;
- emit rõ các video/row lỗi ra `errors.jsonl`, không âm thầm bỏ qua.

Chạy theo khoảng **video**, không cắt một video giữa hai Kaggle run. Cấu hình `VIDEO_START`/`VIDEO_END` áp dụng sau khi sort `video_id`, dạng half-open `[start, end)`. Điều này bảo đảm sequence `_E###` không va chạm giữa các notebook run.

## 3. Contract output bắt buộc

### 3.1 Object R2

Mỗi JPEG mới có object key:

```text
data2/keyframes/<video_id>/<frame_id>.jpg
```

Đây là cách biểu diễn an toàn của thư mục người dùng yêu cầu `/data2/keyframes`; không tạo object key bắt đầu bằng `/` trừ khi cấu hình R2 hiện hữu của người dùng thật sự dùng tiền tố đó.

JPEG phải là RGB, có `ContentType=image/jpeg`. Chỉ coi frame thành công sau khi upload thành công (và nếu API cho phép, `head_object` xác nhận object tồn tại/size khớp).

### 3.2 SQL

Sinh một file duy nhất `insert_keyframes.sql`, chỉ chứa record `source='extracted'`, insert đúng bảng PostgreSQL `frame` (model Python tên `Frame`; tên bảng thực tế là `frame`). Dùng đúng các cột và thứ tự:

```sql
INSERT INTO frame
  (frame_id, n, video_id, shot_id, pts_time, timestamp_ms, fps,
   frame_idx, source, frame_path, width, height)
VALUES (...)
ON CONFLICT (frame_id) DO NOTHING;
```

Quy tắc trường dữ liệu:

- `frame_id`: `<video_id>_E%03d`, sequence tăng liên tục **trên toàn bộ video**, bắt đầu từ `max(existing *_E###) + 1` hoặc 1;
- `n = frame_idx` (đúng module hiện tại);
- `video_id`, `shot_id`, `frame_idx` từ lựa chọn;
- `fps` lấy từ video được probe bằng ffprobe; `pts_time = frame_idx / fps`; `timestamp_ms = round(pts_time * 1000)`;
- `source = 'extracted'`;
- `frame_path = 'data2/keyframes/<video_id>/<frame_id>.jpg'` (object key, không phải URL tự suy đoán);
- `width`, `height` đọc từ JPEG thực tế sau extract.

Escape string SQL đúng chuẩn PostgreSQL (dấu nháy đơn thành `''`), dùng `NULL` thật cho nullable field; không format dữ liệu không tin cậy bằng f-string trực tiếp. Không sinh SQL cho frame upload lỗi, thiếu file, hoặc conflict với keyframe đã có.

Ngoài SQL, sinh `report.jsonl`/`summary.json` để audit (video, shot, selected indices, reason, timings, remote key, upload result). Những file này là artifact benchmark, không thay thế SQL.

## 4. Thuật toán phải giữ nguyên semantics module hiện tại

Sao chép/tái sử dụng logic thuần từ project, nhưng notebook phải self-contained; không phụ thuộc PostgreSQL hoặc đường dẫn local của repo.

### Time baseline và existing frame

- `target_interval_ms=2500`, `min_frame_gap=5`, `max_additional_per_shot=5`.
- Với shot ngắn hơn 2500 ms, target là 1 total keyframe; shot dài dùng `1 + floor(duration / 2500)` nhưng số **bổ sung** không vượt 5.
- Existing frame nằm trong shot được tính vào target. Không chọn frame trùng hoặc quá gần existing frame.

### Hybrid production selector

Mặc định dùng `sentence-transformers/clip-ViT-B-32`, chính xác cùng họ model BTC/project đang dùng. Không tự đổi sang CLIP model khác.

1. Sample candidate bên trong shot ở `1 fps`, tránh biên `transition_margin_frames=2`, tránh existing frame theo `min_frame_gap=5`, giới hạn 64 candidate/shot. Nếu cần giảm candidate thì chọn đều theo thời gian.
2. Lấy existing frame trong shot làm semantic reference, nhưng giới hạn 16 reference/shot, chọn đều theo thời gian. Đây là giới hạn hiệu năng: CLIP/decode tối đa 80 ảnh/shot.
3. Decode candidate/reference bằng PyAV hoặc FFmpeg chính xác theo frame/timestamp. Encode ảnh bằng CLIP, `float32`, L2-normalize.
4. Chọn tối đa 5 candidate bằng deterministic greedy facility-location: ở mỗi vòng chọn frame có marginal gain lớn nhất về cosine coverage của toàn candidate set; reference BTC/đã-extract seed coverage ban đầu. Tie-break bằng `frame_idx` nhỏ hơn. Dừng khi marginal gain `< 0.01`; nếu reference đã cover hoàn toàn shot thì trả về rỗng, không thêm frame chỉ để đủ quota.
5. Lọc low-information bằng HSV histogram 8×8×8. Khi tính bin index phải promote sang `int16`/`int32` trước phép nhân để tránh `uint8` overflow. Dedupe khi cả HSV cosine > 0.8 và CLIP cosine > 0.95.
6. Sau toàn video, cross-shot dedupe ở ranh giới shot liền kề: HSV > 0.75, CLIP > 0.90 và khoảng frame ≤150. Không xoá candidate duy nhất nếu nó tốt hơn candidate bên trái.
7. Nếu CLIP/model/decode hybrid lỗi với một shot, log đầy đủ reason và fallback sang time sampling cho shot đó. Nếu hybrid chạy thành công nhưng chọn rỗng vì existing coverage, **không fallback**.

Kết quả phải deterministic khi cùng input/config/seed.

## 5. Hiệu năng Kaggle và GPU

Ưu tiên GPU Kaggle, nhưng không dùng nhiều process cùng load CLIP vì sẽ giảm VRAM hiệu dụng.

- Kiểm tra CUDA ngay đầu notebook; dùng `device='cuda'` nếu available, `model.eval()` và `torch.inference_mode()`.
- Load `SentenceTransformer` đúng **một lần**. Bắt đầu `CLIP_BATCH_SIZE=128` (cấu hình được), tự giảm một nửa và retry khi OOM; log batch size thực dùng. Không gọi `torch.cuda.empty_cache()` trong loop bình thường.
- Decode/download/upload là I/O/CPU: có thể dùng `ThreadPoolExecutor` bounded (mặc định download 2, upload 8), nhưng mọi encode CLIP chạy ở main GPU worker theo batch. Không share model giữa process.
- Xử lý theo video hoặc nhóm nhỏ shot và giải phóng PIL/NumPy tensors sau nhóm. Không cache toàn bộ RGB của một video. Hard cap 64 candidate + 16 reference/shot.
- Sau khi selection xong cho cả video, export các frame được chọn bằng FFmpeg theo frame index đã sort; dùng select filter single-pass/chunk, `FFMPEG_EXPORT_CHUNK_SIZE=100`. Chỉ fallback per-frame nếu single-pass thất bại và phải log fallback.
- Download video streaming vào `/kaggle/working`; kiểm tra file không rỗng và ffprobe được trước xử lý. Dọn file video/JPEG tạm chỉ sau khi upload và checkpoint thành công.
- Bật `torch.backends.cudnn.benchmark=True` cho image shape ổn định; seed NumPy/Python/Torch để reproducible.

Không khẳng định GPU tăng tốc FFmpeg decoding nếu Kaggle runtime không có khả năng đó. GPU được ưu tiên cho CLIP; decode/export dùng FFmpeg CPU trừ khi runtime đã xác minh được NVDEC/NVJPEG an toàn.

## 6. Cloudflare R2 upload

Người dùng sẽ cung cấp cấu hình/secret. Dùng Kaggle Secrets hoặc environment variables, không hard-code và không in secret:

```text
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_BUCKET=<bucket>
R2_ACCESS_KEY_ID=<secret>
R2_SECRET_ACCESS_KEY=<secret>
R2_KEY_PREFIX=data2/keyframes
```

Dùng client S3-compatible (`boto3`) với `region_name='auto'`, endpoint cấu hình được. Upload có retry exponential backoff, bounded concurrency, content type JPEG và `head_object` verification. Nếu object đã có, kiểm tra size/metadata trước; chỉ skip khi khớp và checkpoint hợp lệ. Không upload partial/corrupt file, không xoá object remote của run trước.

Giữ secret ở `os.environ`/Kaggle Secrets; `.env.example` chỉ chứa tên biến và không được xuất secret ra SQL/report/notebook output.

## 7. Checkpoint, retry và tính nhất quán

Lưu checkpoint atomic trong `/kaggle/working` sau từng video thành công, ít nhất gồm: input/config hash, completed video IDs, selected/uploaded rows, remote keys và SQL fragment/checksum. Khi restart:

- reject checkpoint nếu input/config hash không khớp trừ khi người dùng bật explicit override;
- skip video đã hoàn tất;
- retry các frame chưa upload;
- chỉ append SQL record sau khi remote object verified;
- cuối run ghép/chọn lại SQL theo thứ tự `(video_id, shot_index, frame_idx)` để output deterministic.

Không connect hoặc execute SQL trên database từ Kaggle.

## 8. Cell test bắt buộc

Thêm cell mock test cuối notebook, không cần dữ liệu BTC:

- tạo video MP4 ngắn bằng FFmpeg với nhiều cảnh màu/texture;
- tạo `shot.csv`, `keyframe.csv`, `videos.csv` nhỏ;
- mock R2 client (không gọi cloud thật);
- chạy pipeline hai lần;
- assert: không trùng `frame_idx`, không vượt 5 additional/shot, không ghi đè `_E###`, official frame được dùng làm reference, SQL chỉ có uploaded extracted records, object key đúng `data2/keyframes/...`, và lần 2 idempotent.

## 9. Tiêu chí bàn giao

Notebook hiển thị summary gồm số video/shot, official frame, selected/extracted/uploaded/failed frame, fallback count, thời gian download/decode/CLIP/export/upload, GPU name/peak VRAM nếu lấy được. Nêu rõ bất kỳ dependency Kaggle nào cần cài. Không thay đổi schema DB, không thay model CLIP BTC, không phát sinh caption/ASR.
