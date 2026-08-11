# Runbook cho AI agent: dựng VM và chạy main pipeline từ A đến Z

Tài liệu này là quy trình thao tác cho một AI agent nhận một VM Ubuntu mới và
một repository vừa clone. Mục tiêu cuối cùng là tải đủ AIC25 B1, dựng đúng môi
trường GPU/PostgreSQL, nạp dữ liệu organizer, rồi chạy
`BackEnd.app.pipeline.main_pipeline` với PostgreSQL và FAISS thật.

Không dùng tài liệu này để sửa hoặc chạy lại trên database/FAISS production đã
có dữ liệu. Pipeline hiện tại chưa bảo đảm idempotent cho toàn bộ output.

## 1. Quy tắc bắt buộc cho agent

1. Đọc `AGENTS.md` (nếu có), `agent.md`, `BackEnd/CONFIG.py` và tài liệu này
   trước khi chạy lệnh.
2. Báo cho người dùng trước khi cài system package, tạo database hoặc bắt đầu
   job dài.
3. Không in, log hoặc commit mật khẩu và nội dung `.env`.
4. Không xóa dataset, database, FAISS index, model hoặc Conda environment nếu
   chưa được người dùng phê duyệt rõ ràng.
5. Chạy mọi lệnh Python trong Conda environment `DL_Env`.
6. Sau mỗi nhóm thao tác, chạy lệnh kiểm chứng tương ứng. Không tuyên bố thành
   công chỉ dựa trên việc process còn chạy.
7. Nếu một lệnh lỗi, giữ nguyên log và sửa nguyên nhân. Không blind-retry một
   pipeline ghi dữ liệu vì có thể gây trùng khóa DB/FAISS.

## 2. Phạm vi pipeline hiện tại

`main_pipeline.py` hiện chạy các stage sau:

```text
Shot extraction
    -> Additional keyframe extraction
    -> Logical clip extraction
    -> OCR
    -> YOLO26 + ByteTrack tracking
    -> Extracted-frame, clip và shot embeddings
```

Tracking và nhánh keyframe/clip/OCR có thể chạy song song khi GPU preflight cho
phép. ObjectDetection organizer là nguồn độc lập và được nạp trước từ JSONL;
main pipeline không chạy lại Faster R-CNN. ASR và caption chưa được nối vào
`main_pipeline.py`, vì vậy không được báo rằng hai stage này đã chạy.

## 3. Yêu cầu VM

- Ubuntu 22.04 hoặc 24.04 x86_64.
- GPU mục tiêu: RTX A6000 48GB. Driver phải hỗ trợ CUDA 12.6 wheels.
- RAM hệ thống: tối thiểu 32GB, khuyến nghị 64GB.
- Disk trống: tối thiểu 200GB, khuyến nghị 500GB.
- Dataset đã giải nén chiếm khoảng 110GB: video 78GB, keyframe 30GB, object
  JSON 2.1GB và các phần còn lại khoảng 1GB. PostgreSQL/WAL, model cache, FAISS,
  extracted frames và artifacts cần thêm dung lượng. Disk 100GB không đủ.
- Network ổn định; downloader dùng nhiều connection và hỗ trợ resume.

Kiểm tra trước:

```bash
nvidia-smi
df -h
free -h
uname -a
```

Nếu GPU không đúng hoặc disk trống dưới 200GB, dừng và báo người dùng.

## 4. Cài công cụ hệ thống

Sau khi được cho phép:

```bash
sudo apt update
sudo apt install -y git curl ffmpeg aria2 unzip rsync tmux postgresql postgresql-contrib libgl1 libglib2.0-0
sudo systemctl enable --now postgresql
```

Kiểm chứng:

```bash
git --version
ffmpeg -version
ffprobe -version
aria2c --version
unzip -v
rsync --version
psql --version
systemctl is-active postgresql
```

## 5. Clone đúng revision

```bash
git clone <REPOSITORY_URL> 4US-1F-AIC-HCMC
cd 4US-1F-AIC-HCMC
git fetch --all --prune
git checkout <BRANCH_OR_COMMIT_APPROVED_BY_USER>
git rev-parse HEAD
git status --short
```

Ghi lại commit hash vào báo cáo triển khai. Không tự merge hoặc đổi branch.

## 6. Cài Miniconda và tạo `DL_Env`

Nếu `~/miniconda3` chưa tồn tại:

```bash
curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
```

Khởi tạo environment mới chỉ khi `DL_Env` chưa tồn tại:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env list
conda create -n DL_Env python=3.12 -y
conda activate DL_Env
which python
python --version
```

Kết quả phải là Python trong `.../miniconda3/envs/DL_Env/` và Python 3.12.

## 7. Cài dependency theo thứ tự đã kiểm chứng

Không cài `requirements-object-detection.txt` vào `DL_Env`. TensorFlow CUDA và
Paddle GPU có pin NVIDIA runtime xung đột; Faster R-CNN không được dùng trong
main pipeline hiện tại.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-paddle-gpu.txt
python -m pip list | grep -E '^paddlepaddle([[:space:]]|-gpu)'
```

Kết quả cuối chỉ được có `paddlepaddle-gpu`; nếu đồng thời có package
`paddlepaddle` CPU thì dừng và báo người dùng, không tự uninstall môi trường đã
có.

Lý do Paddle được cài `--no-deps`: bản đã kiểm chứng dùng
`torch==2.7.0+cu126`, cần `nvidia-nccl-cu12==2.26.2`, trong khi
`paddlepaddle-gpu==3.3.0` khai báo đúng bằng `2.25.1`. Pipeline chỉ inference
trên một GPU và không dùng NCCL distributed. Runtime thực tế đã chạy được cả
Torch CUDA và Paddle CUDA với NCCL do Torch cung cấp.

`python -m pip check` được phép báo đúng một metadata warning về NCCL nói trên.
Bất kỳ lỗi dependency nào khác đều phải được điều tra trước khi tiếp tục:

```bash
python -m pip check
```

Kiểm chứng bằng phép tính thật trên GPU, không chỉ import:

```bash
python -c "import torch; x=torch.ones(4, device='cuda'); print('Torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), x.sum().item())"
python -c "import paddle; paddle.set_device('gpu:0'); x=paddle.ones([4]); print('Paddle', paddle.__version__, paddle.device.is_compiled_with_cuda(), float(x.sum()))"
```

Hai lệnh phải hoàn tất trên GPU. Không chấp nhận CPU fallback.

## 8. Tải đầy đủ dataset AIC25 B1

Nên chạy trong `tmux` để SSH mất kết nối không làm dừng tải:

```bash
tmux new -s aic-data
cd <PROJECT_ROOT>
./scripts/download_aic25_data.sh --jobs 2 --connections 16
```

Detach bằng `Ctrl+B`, rồi `D`. Vào lại bằng:

```bash
tmux attach -t aic-data
```

Script chứa đủ 32 URL trong spreadsheet công khai:

- 14 archive keyframe từ L21 đến L30;
- 14 archive video từ L21 đến L30;
- `clip-features-32-aic25-b1.zip` (`.npy`);
- `map-keyframes-aic25-b1.zip` (`.csv`);
- `media-info-aic25-b1.zip` (`.json` metadata);
- `objects-aic25-b1.zip` (`.json` object detection).

Mỗi ZIP được kiểm tra CRC bằng `unzip -tq` trước khi extract. File tải dở được
resume bằng aria2c. Sau khi extract thành công, script tạo completion marker
trong `data/.downloads/.completed/`, vì vậy chạy lại sẽ không tải lại archive
đã hoàn thành. Không xóa marker hoặc dùng `--overwrite` nếu chưa xác định dữ
liệu nào thực sự hỏng.

Tổng connection là `jobs * connections`. Với đường truyền 1Gbps, bắt đầu bằng
2 x 16. Chỉ tăng khi CDN và disk vẫn đáp ứng; không vượt 64 connection tổng.

Script tự verify sau một lượt tải toàn bộ. Có thể chạy riêng:

```bash
./scripts/download_aic25_data.sh --verify-only
```

Kết quả bắt buộc:

```text
videos (.mp4)            873
keyframes (.jpg)          >= 177321
keyframe maps (.csv)      873
metadata (.json)          873
CLIP features (.npy)      873
object files (.json)      177321
```

Video ID của video, keyframe directory, map, metadata, NPY và object directory
phải khớp nhau. Cấu trúc cuối cùng phải là:

```text
data/video/<video_id>.mp4
data/keyframes/<video_id>/<n:03d>.jpg
data/map-keyframes/<video_id>.csv
data/clip-features-32/<video_id>.npy
data/media-info-aic25-b1/media-info/<video_id>.json
data/objects-aic25-b1/objects/<video_id>/<n:03d>.json
```

Hai path cuối là nơi từng bị giải nén sai. Không chấp nhận các cấu trúc
`.../media-info/media-info/`, `.../objects/objects/` hoặc file nằm trực tiếp ở
`data/media-info-aic25-b1/`.

## 9. Chuẩn bị model và cache

Tạo thư mục model:

```bash
mkdir -p data/models
```

YOLO26n chính thức có thể được Ultralytics tải tự động:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
(cd data/models && python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); print('YOLO26n ready')")
test -s data/models/yolo26n.pt
```

TransNetV2 cần checkpoint PyTorch đã convert đúng với implementation của repo:

```text
data/models/transnetv2-pytorch-weights.pth
```

Checkpoint này không nằm trong Git và không nằm trong spreadsheet dataset.
Agent phải yêu cầu người dùng cung cấp artifact/checksum hoặc copy từ máy đã
được xác minh. Không tự dùng một file cùng tên từ nguồn không rõ. Sau khi copy:

```bash
test -s data/models/transnetv2-pytorch-weights.pth
sha256sum data/models/transnetv2-pytorch-weights.pth
```

Warm-up OCR và CLIP để thời gian download model không lẫn vào pipeline:

```bash
python -c "from BackEnd.app.ocr.service import OCRService; s=OCRService(); s.engine; print('OCR ready'); s.close()"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cuda'); print('CLIP ready')"
```

Cảnh báo `No ccache found` từ Paddle không phải lỗi inference. Không cần cài
`ccache` chỉ để bỏ cảnh báo này.

## 10. Tạo PostgreSQL và `.env`

Dùng database mới, ví dụ `aic_pipeline`. Không chạy trên database production
cũ. Tạo role bằng prompt để mật khẩu không nằm trong command history:

```bash
sudo -u postgres createuser --pwprompt aic_pipeline_user
sudo -u postgres createdb --owner=aic_pipeline_user aic_pipeline
```

Tạo `.env` tại project root với permission 600. Agent phải nhận secret từ
người dùng và không hiển thị lại. Nếu mật khẩu chứa ký tự đặc biệt, URL-encode
phần password:

```text
DATABASE_URL=postgresql+psycopg://aic_pipeline_user:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/aic_pipeline
```

```bash
chmod 600 .env
git status --short
```

`.env` phải được Git ignore và không xuất hiện trong `git status`.

Với database mới, chỉ dùng SQLAlchemy models hiện tại để tạo schema; không chạy
`postgre_script.sql` rồi lại chạy `init_db`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
python -c "from BackEnd.app.database.postgre_db import PostgreManager; db=PostgreManager(); db.init_db(); db.engine.dispose(); print('Schema ready')"
```

Các SQL trong `BackEnd/app/database/migrations/` chỉ dành cho database cũ cần
nâng schema và phải backup/review riêng. Không áp migration đó vào database mới.

## 11. Convert object JSON sang JSONL

Chạy đúng một lần trên dữ liệu organizer. Threshold mặc định là `>= 0.25`:

```bash
python -m BackEnd.app.object_detection.convert_aic25_objects_to_jsonl
```

Kết quả đã kiểm chứng trên full dataset:

```text
Converted 177321 files and retained 1332453 detections.
```

Output phải nằm ở:

```text
data/objects-aic25-b1-jsonl/objects/<video_id>/<n:03d>.jsonl
```

Không dùng `--overwrite` trừ khi người dùng yêu cầu tạo lại toàn bộ JSONL.

## 12. Nạp dữ liệu organizer theo đúng thứ tự

Tất cả lệnh sau ghi thật vào PostgreSQL/FAISS. Dùng cùng một database mới và
cùng một FAISS directory trong toàn bộ quy trình.

### 12.1 Video metadata

```bash
python -m BackEnd.app.pipeline.video_ingestion
```

Phải nạp 873 video. Việc description dài bị truncate theo schema có thể tạo
warning nhưng không phải lỗi.

### 12.2 ClassID

```bash
python -m BackEnd.app.pipeline.class_id_ingestion
```

Phải nạp 601 class trước ObjectDetection và Tracking. Nếu bỏ bước này sẽ gặp
foreign-key violation ở `objecttrack.class_id`.

### 12.3 Official Frame metadata

Chạy validation không ghi DB trước:

```bash
python -m BackEnd.app.pipeline.official_keyframe_ingestion --dry-run
python -m BackEnd.app.pipeline.official_keyframe_ingestion
```

Phải khám phá 177.321 logical official frames thuộc 873 video.

### 12.4 ObjectDetection organizer

```bash
python -m BackEnd.app.pipeline.object_detection_ingestion
```

Phải nạp 1.332.453 detection từ JSONL. Module này độc lập với Tracking và không
được thay bằng YOLO tracking output.

### 12.5 Official frame embeddings

```bash
mkdir -p artifacts/faiss
python -m BackEnd.app.pipeline.embedding_ingestion --faiss-dir artifacts/faiss
```

Phải nạp 177.321 mapping và vector official vào frame FAISS index. Main pipeline
cố định dùng cùng `artifacts/faiss`; đổi index trong khi DB vẫn có mapping cũ sẽ
gây trùng `faiss_id` hoặc mất đồng bộ.

## 13. Kiểm tra code và dữ liệu trước khi chạy main pipeline

```bash
./scripts/download_aic25_data.sh --verify-only
test -s data/models/yolo26n.pt
test -s data/models/transnetv2-pytorch-weights.pth
nvidia-smi
```

Chạy test tập trung:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
pytest -q BackEnd/tests/pipeline/test_main_pipeline.py BackEnd/tests/pipeline/test_ocr_pipeline.py BackEnd/tests/pipeline/test_tracking_pipeline.py
```

Trên A6000, parallel preflight cần ít nhất 36GB VRAM trống: 12GB tracking,
16GB OCR và 8GB headroom. Cấu hình mặc định `auto` sẽ tự chuyển sang tuần tự
nếu không đủ VRAM.

## 14. Chạy main pipeline

Chạy job dài trong `tmux` và ghi log:

```bash
tmux new -s aic-pipeline
cd <PROJECT_ROOT>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
python -m BackEnd.app.pipeline.main_pipeline 2>&1 | tee artifacts/main_pipeline.log
```

Lệnh này luôn xử lý toàn bộ video đã có trong PostgreSQL, dùng
`PIPELINE_PARALLEL_MODE` trong `BackEnd/CONFIG.py` và dùng chung FAISS directory
`artifacts/faiss`. Main pipeline không cung cấp option chọn riêng video hoặc
đổi runtime mode/path từ command line.

Theo dõi ở SSH session khác:

```bash
tmux attach -t aic-pipeline
watch -n 1 nvidia-smi
tail -f artifacts/main_pipeline.log
df -h
```

`Ctrl+C` chỉ thoát `watch`/`tail`; không dừng job trong tmux khác. Không kill
pipeline trừ khi người dùng yêu cầu hoặc có OOM/disk-full gây mất an toàn.

## 15. Tiêu chí hoàn tất

Chỉ báo pipeline thành công khi có đủ các bằng chứng sau:

1. Process kết thúc với exit code 0 và log có
   `Pipeline completed for <N> video(s).`.
2. Không có traceback, CUDA OOM, foreign-key violation hoặc duplicate FAISS ID.
3. PostgreSQL có Shot, extracted Frame, ClipWindow, OCR, ObjectTrack,
   TrackObservation và embedding mappings.
4. `artifacts/faiss/frame.faiss`, `clip.faiss`, `shot.faiss` và
   `faiss_index.json` tồn tại, có kích thước lớn hơn 0.
5. Disk chưa gần đầy và GPU không còn process pipeline bị orphan.
6. `git status --short` không chứa `.env`, dataset, weights, FAISS hay generated
   artifacts.

Kiểm tra artifact:

```bash
find artifacts/faiss -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
ps -ef | grep '[B]ackEnd.app.pipeline.main_pipeline' || true
nvidia-smi
git status --short
```

## 16. Lỗi thường gặp

- `No ccache found`: warning của Paddle, không phải nguyên nhân job đứng. Kiểm
  tra process và `nvidia-smi`; GPU utilization cao nghĩa là đang xử lý.
- `Skipping invalid OCR polygon`: một OCR polygon không hợp lệ bị bỏ qua; không
  fatal. Điều tra nếu gần như mọi polygon bị bỏ hoặc OCR table rỗng.
- `objecttrack_class_id_fkey`: chưa nạp đủ ClassID hoặc DB schema/data sai thứ
  tự. Chạy/kiểm tra class ingestion, không bỏ foreign key.
- Duplicate `(faiss_id, index_version)`: DB mapping và FAISS directory không
  cùng trạng thái. Không tạo index mới rồi dùng lại DB cũ.
- CUDA OOM ở parallel mode: giữ log, đặt `PIPELINE_PARALLEL_MODE = "sequential"`
  trong `BackEnd/CONFIG.py` hoặc giảm batch size có chủ đích; không tăng worker.
- `No such file or directory` cho metadata/object: kiểm tra hai path lồng đúng
  ở mục 8 và chạy `--verify-only`.
- Download gián đoạn: chạy lại đúng command. Aria2 resume file dở và completion
  markers bỏ qua archive đã extract thành công.
- SSH ngắt: dùng `tmux`; không khởi chạy job thứ hai trước khi kiểm tra process
  cũ.

## 17. Báo cáo cuối cho người dùng

Agent phải báo ngắn gọn:

- commit hash đã triển khai;
- GPU/driver, RAM và disk;
- kết quả dataset verification và các count;
- phiên bản Torch/Paddle và kết quả CUDA runtime;
- database/FAISS logical name, tuyệt đối không ghi credential;
- các ingestion count;
- chế độ pipeline thực tế (`parallel` hay fallback `sequential`);
- thời gian chạy, exit code và đường dẫn log;
- stage chưa nằm trong main pipeline (ASR/caption);
- mọi giới hạn hoặc bước chưa thể xác minh.
