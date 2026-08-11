# Checklist dựng VM và chạy benchmark pipeline

Mục tiêu là dựng một VM staging có thể chạy đúng pipeline hiện tại và benchmark
ghi thật vào PostgreSQL + FAISS trong khoảng một giờ. Không dùng database,
FAISS index hoặc thư mục output của môi trường production.

Checklist này không bao gồm việc tải dữ liệu dataset. Giả sử dữ liệu đã được
mount/copy vào `<PROJECT_ROOT>/data` với đúng cấu trúc hiện có.

## 0. Quy ước an toàn

- [ ] VM benchmark dùng database riêng, ví dụ `aic_benchmark`; không dùng
  database production.
- [ ] Chọn ít nhất một `video_id` chưa có output pipeline (`Shot`, Frame
  extracted, ClipWindow, OCR, ObjectTrack, embedding mapping) trong database
  benchmark. Pipeline hiện không phải một job re-run idempotent cho các output
  này.
- [ ] Giữ riêng FAISS index benchmark tại `artifacts/benchmark/.../faiss`.
  Chỉ truyền `--faiss-dir` nếu chủ đích kiểm tra index đã tồn tại trên VM test.
- [ ] Không commit `.env`, model weights, FAISS index, artifact benchmark hoặc
  dữ liệu vào Git.

## 1. Chọn và kiểm tra VM

- [ ] Ubuntu 22.04/24.04, GPU RTX A6000 48GB, SSD/NVMe và tối thiểu 100GB trống.
  Khuyến nghị 200GB nếu PostgreSQL, WAL và model cache cùng nằm trên VM.
- [ ] Đủ RAM hệ thống: tối thiểu 32GB, khuyến nghị 64GB.
- [ ] CUDA driver nhận GPU:

  ```bash
  nvidia-smi
  ```

  Kết quả phải hiển thị A6000, 48GB VRAM và không có process lạ chiếm nhiều
  VRAM.

- [ ] Kiểm tra disk trước khi copy model/data:

  ```bash
  df -h
  ```

## 2. Cài công cụ hệ thống

- [ ] Cài Git, FFmpeg/FFprobe cho decode video, PostgreSQL client/server và các
  shared library thường cần cho OpenCV/Pillow:

  ```bash
  sudo apt update
  sudo apt install -y git ffmpeg postgresql postgresql-contrib libgl1 libglib2.0-0
  ```

- [ ] Xác nhận:

  ```bash
  git --version
  ffmpeg -version
  ffprobe -version
  psql --version
  ```

## 3. Clone repository và tạo Conda environment

- [ ] Clone đúng branch/commit cần benchmark:

  ```bash
  git clone <REPOSITORY_URL> <PROJECT_ROOT>
  cd <PROJECT_ROOT>
  git rev-parse HEAD
  ```

- [ ] Cài Miniconda nếu VM chưa có. Tạo environment Python 3.12 tên `DL_Env`:

  ```bash
  conda create -n DL_Env python=3.12 -y
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate DL_Env
  python --version
  ```

- [ ] Cài requirements. Lệnh này tải dependency; không chạy khi chưa có network
  hoặc khi chưa sẵn sàng thay package Paddle CPU bằng Paddle GPU:

  ```bash
  python -m pip uninstall paddlepaddle
  python -m pip install -r requirements.txt
  ```

  `requirements.txt` dùng `paddlepaddle-gpu==3.3.0` từ channel CUDA 12.6.
  Không cài đồng thời `paddlepaddle` và `paddlepaddle-gpu`.

## 4. Kiểm tra runtime GPU trước khi tải model

- [ ] PyTorch và Paddle đều phải thấy CUDA. Nếu một trong hai là `False`, không
  chạy benchmark parallel:

  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate DL_Env
  python -c "import torch, paddle; print('PyTorch:', torch.__version__, torch.cuda.is_available(), torch.version.cuda); print('Paddle:', paddle.__version__, paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
  ```

- [ ] Xác nhận đúng GPU từ PyTorch:

  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GiB')"
  ```

- [ ] Xác nhận không có mismatch dependency quan trọng:

  ```bash
  python -m pip check
  ```

## 5. Chuẩn bị dữ liệu và model weights

- [ ] Bảo đảm các thư mục dữ liệu cần cho benchmark tồn tại:

  ```bash
  test -d data/video
  test -d data/media-info-aic25-b1/media-info
  test -d data/keyframes
  test -d data/map-keyframes
  test -d data/objects-aic25-b1
  test -d data/objects-aic25-b1-jsonl/objects
  test -d data/clip-features-32
  ```

- [ ] Copy model tracking và TransNet vào đúng path cấu hình:

  ```bash
  test -f data/models/yolo26n.pt
  test -f data/models/transnetv2-pytorch-weights.pth
  ```

- [ ] Nếu muốn tránh download CLIP trong lúc benchmark, copy model đã cache vào:

  ```text
  data/models/clip-ViT-B-32/
  ```

  Nếu không có thư mục này, `sentence-transformers` sẽ tải
  `clip-ViT-B-32` ở lần embedding đầu tiên; download đó không nên tính vào
  benchmark.

- [ ] OCR dùng Paddle detector và VietOCR. Khởi tạo OCR một lần trước benchmark
  để tải/cache weights, sau đó kiểm tra `nvidia-smi` không có OOM. Chỉ cần import
  và tạo service; không ghi database:

  ```bash
  python -c "from BackEnd.app.ocr.service import OCRService; service = OCRService(); service.engine; print('OCR model warm-up completed'); service.close()"
  ```

- [ ] Không cần chuẩn bị Faster R-CNN/Open Images TensorFlow weights cho
  benchmark visual hiện tại: ObjectDetection được nạp từ JSONL, không infer lại.
- [ ] ASR ChunkFormer và caption Qwen2-VL không nằm trong `main_pipeline.py` và
  không thuộc benchmark này. Chuẩn bị/chạy chúng thành benchmark riêng sau đó.

## 6. Khởi tạo PostgreSQL staging

- [ ] Tạo role/database riêng. Thay placeholder, không ghi mật khẩu vào shell
  history hoặc Git:

  ```bash
  sudo -u postgres createuser --pwprompt aic_benchmark_user
  sudo -u postgres createdb --owner=aic_benchmark_user aic_benchmark
  ```

- [ ] Tạo `.env` ở project root, quyền chỉ cho user hiện tại:

  ```bash
  umask 077
  printf '%s\n' 'DATABASE_URL=postgresql+psycopg://aic_benchmark_user:YOUR_PASSWORD@127.0.0.1:5432/aic_benchmark' > .env
  ```

- [ ] Tạo schema bằng SQLAlchemy từ models hiện tại:

  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate DL_Env
  python -c "from BackEnd.app.database.postgre_db import PostgreManager; db = PostgreManager(); db.init_db(); db.engine.dispose(); print('Schema created')"
  ```

- [ ] Kiểm tra kết nối và các bảng cơ bản:

  ```bash
  python -c "from BackEnd.app.database.postgre_db import PostgreManager; db = PostgreManager(); print(len(db.get_list_video())); db.engine.dispose()"
  ```

## 7. Nạp dữ liệu nền vào PostgreSQL theo đúng thứ tự

Các bước này ghi staging DB thật. Có thể giới hạn `--video-id L23_V005` ở
official keyframe/object/embedding khi chỉ benchmark một video.

- [ ] Nạp Video từ metadata:

  ```bash
  python -m BackEnd.app.pipeline.video_ingestion
  ```

- [ ] Nạp toàn bộ 601 `ClassID`. Đây là bắt buộc trước Tracking và
  ObjectDetection để tránh lỗi foreign key:

  ```bash
  python -m BackEnd.app.pipeline.class_id_ingestion
  ```

- [ ] Nạp official Frame metadata. Chạy dry-run trước nếu nghi ngờ path dữ liệu:

  ```bash
  python -m BackEnd.app.pipeline.official_keyframe_ingestion --video-id L23_V005 --dry-run
  python -m BackEnd.app.pipeline.official_keyframe_ingestion --video-id L23_V005
  ```

- [ ] Nếu benchmark cần cả nguồn ObjectDetection, nạp JSONL sau Frame và
  ClassID:

  ```bash
  python -m BackEnd.app.pipeline.object_detection_ingestion --video-id L23_V005
  ```

- [ ] Nếu benchmark cần official CLIP vectors, nạp vào FAISS trước khi chạy
  extracted-frame embedding. Với FAISS benchmark riêng, truyền cùng directory
  vào lệnh ingest và benchmark:

  ```bash
  python -m BackEnd.app.pipeline.embedding_ingestion \
    --video-id L23_V005 \
    --faiss-dir artifacts/benchmark/bootstrap-faiss
  ```

  Nếu bỏ bước này, benchmark vẫn đo được extracted frame/clip/shot embedding;
  chỉ thiếu vector official organizer trong index benchmark.

## 8. Kiểm tra trước benchmark

- [ ] Xác nhận video benchmark tồn tại và `Video` record đã được nạp:

  ```bash
  test -f data/video/L23_V005.mp4
  python -c "from BackEnd.app.database.postgre_db import PostgreManager; db = PostgreManager(); print([v.video_id for v in db.get_list_video() if v.video_id == 'L23_V005']); db.engine.dispose()"
  ```

- [ ] Kiểm tra VRAM trống tối thiểu 36GB. Parallel scheduler dành 12GB Tracking,
  16GB OCR và 8GB safety headroom:

  ```bash
  nvidia-smi
  ```

- [ ] Chạy test code không cần GPU trước:

  ```bash
  pytest -q BackEnd/tests/pipeline/test_main_pipeline.py BackEnd/tests/pipeline/test_ocr_pipeline.py BackEnd/tests/pipeline/test_tracking_pipeline.py
  ```

- [ ] Nếu database benchmark đã từng chạy cùng video, tạo database staging mới
  hoặc dùng video khác. Không truncate/xóa dữ liệu production.

## 9. Chạy benchmark ghi thật trong một giờ

Lệnh này buộc scheduler chạy Tracking song song nhánh Keyframe/Clip/OCR. Nó ghi
vào PostgreSQL benchmark và tạo FAISS index benchmark cô lập:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
python -m BackEnd.app.pipeline.benchmark_pipeline \
  --video-id L23_V005 \
  --duration-seconds 3600 \
  --parallel-mode parallel
```

Lưu ý: `--duration-seconds` là giới hạn mềm tại ranh giới stage. Script không
kill một stage/transaction đang chạy chỉ để đúng 3,600 giây, vì làm vậy có thể
để database hoặc FAISS ở trạng thái không nhất quán.

Nếu cần benchmark nhiều video, lặp `--video-id`:

```bash
python -m BackEnd.app.pipeline.benchmark_pipeline \
  --video-id L23_V005 \
  --video-id L21_V001 \
  --duration-seconds 3600 \
  --parallel-mode parallel
```

## 10. Đọc kết quả và quyết định

- [ ] Mở report mới nhất:

  ```bash
  find artifacts/benchmark -name report.json -print | sort | tail -n 1
  ```

- [ ] Report phải có:

  - `error: null`;
  - `preflight.parallel_safe: true`;
  - thời gian của `shot_extraction`, `parallel_processing`, và nếu đủ thời gian,
    `embeddings`;
  - `max_memory_used_mib` thấp hơn 40GiB để còn headroom;
  - GPU utilization không bằng 0 trong tracking/OCR.

- [ ] Xem mẫu GPU khi cần điều tra peak VRAM hoặc GPU idle:

  ```bash
  tail -n 20 artifacts/benchmark/<RUN_ID>/gpu_samples.jsonl
  cat artifacts/benchmark/<RUN_ID>/report.json
  ```

- [ ] Nếu OOM, giữ report/log, giảm `OCR_DETECTION_BATCH_SIZE`,
  `OCR_RECOGNITION_BATCH_SIZE` hoặc `EMBEDDING_BATCH_SIZE`; không tăng số worker
  vượt hai.
- [ ] Nếu GPU utilization thấp nhưng không OOM, kiểm tra `iostat`, CPU, decode
  FFmpeg/PyAV và PostgreSQL/WAL trước khi đổi model hoặc thuê GPU mạnh hơn.
- [ ] Chỉ sau benchmark pass mới đồng bộ cùng model/data/config sang VM chính.
