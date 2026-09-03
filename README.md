# AIC HCMC Video Preprocessing

Pipeline **offline** để trích xuất, làm giàu và chuẩn bị dữ liệu video cho bài toán video retrieval của AIC HCMC. Repo không phải ứng dụng web và không phục vụ truy vấn trực tiếp. Đầu vào là video cùng metadata/dữ liệu do BTC cung cấp; đầu ra là các artifact có thể kiểm tra, gồm CSV/JSONL, FAISS index và SQL `INSERT` để đội vận hành nạp vào PostgreSQL theo thứ tự kiểm soát.

Pipeline hỗ trợ:

- tách video thành shot bằng TransNetV2;
- chọn keyframe bổ sung theo thời gian hoặc chiến lược hybrid CLIP;
- tạo cửa sổ clip chồng lấp từ shot;
- OCR tiếng Việt, object detection, và theo dõi đối tượng bằng YOLO + ByteTrack;
- tiền xử lý audio và ASR;
- tạo embedding frame, clip và shot để xây dựng FAISS index;
- tạo FrameContext từ Caption/OCR/Object và dense text index cho Context/ASR;
- chuyển dữ liệu object chính thức của BTC thành SQL.

## Nguyên tắc vận hành

- Xử lý theo batch, có thể resume/checkpoint tùy stage; không có API runtime.
- Notebook Kaggle chỉ đọc Dataset đầu vào và ghi vào `/kaggle/working`; **không** tự kết nối PostgreSQL hoặc thực thi SQL.
- SQL sinh ra là artifact để DBA/importer kiểm duyệt và nạp sau, không được chạy lặp lại một cách mặc định.
- Các mô hình và trọng số nên được chuẩn bị cục bộ/đính kèm Dataset trước khi chạy full; pipeline không được thiết kế để tự tải dữ liệu production.
- Không trộn FAISS ID, model revision hoặc embedding artifact giữa các lần chạy khác nhau.

## Kiến trúc dữ liệu

```text
video + metadata
       |
       v
Shot Extractor ───────> shot.csv / insert_shots.sql
       |\
       | +--> Keyframe Extractor ──> keyframe.csv + ảnh + insert_keyframes.sql
       | |          |                         |
       | |          +--> OCR ────────────────> insert_ocr.sql
       | |          +--> Frame Embedding ────> FAISS + insert_frame_embedding_records.sql
       | |
       | +--> Clip Extractor ────────────────> clipwindow.csv + insert_clipwindows.sql
       | |          |
       | |          +--> Clip Embedding ─────> FAISS + insert_clip_embedding_records.sql
       | |                                      |
       | |                                      +--> Shot Embedding ─> FAISS + insert_shot_embedding_records.sql
       | |
       | +--> Tracking ──────────────────────> insert_tracking.sql
       |
BTC object JSONL ──────────────────────────────> insert_object_detection.sql
```

Caption và ASR là stage bổ sung. Caption gọi API FPT nên không phải luồng offline hoàn toàn; ASR chạy từ audio đã chuẩn hóa.

## Thành phần chính

| Thành phần | Vai trò | Đầu ra chính |
| --- | --- | --- |
| `BackEnd/app/shot_extractor/` | Phát hiện ranh giới shot bằng TransNetV2 | `shot` |
| `BackEnd/app/keyframe_extractor/` | Trích keyframe bổ sung, tránh frame dư thừa | `frame`, ảnh keyframe |
| `BackEnd/app/clip_extractor/` | Sinh clip window 10 giây, stride 8 giây mặc định | `clipwindow` |
| `BackEnd/app/ocr/` | PP-OCRv5 detection + VietOCR/PaddleOCR recognition | `ocr` |
| `BackEnd/app/object_detection/` | YOLO hoặc chuyển JSONL Open Images của BTC | `objectdetection` |
| `BackEnd/app/tracking/` | YOLO26 + ByteTrack, reset tracker theo shot | `objecttrack`, `trackobservation` |
| `BackEnd/app/embedding/` | SigLIP/CLIP embedding, artifact và FAISS | embedding records + index |
| `BackEnd/app/frame_context/` | Ghép Caption, OCR và Object thành text evidence theo frame | FrameContext Parquet + manifest |
| `BackEnd/app/text_embedding/` | Dense embedding và FAISS cho FrameContext/ASR segment | index + mapping + manifest |
| `BackEnd/app/audio_pre/`, `BackEnd/app/ASR/` | Chuẩn hóa audio theo shot và nhận dạng tiếng nói | audio artifact, transcript |
| `BackEnd/app/database/` | PostgreSQL, Elasticsearch và FAISS adapters | persistence/search index |
| `Notebook/` | Các notebook Kaggle độc lập, xuất SQL/artifact | artifact theo từng stage |

Các schema dùng chung nằm tại `BackEnd/app/contracts/`. Cấu hình tập trung tại [BackEnd/CONFIG.py](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/BackEnd/CONFIG.py).

## Yêu cầu môi trường

- Ubuntu x86_64
- Python 3.12 (`>=3.12,<3.13`)
- `uv`
- NVIDIA GPU + CUDA 12.6 cho các stage GPU chính
- FFmpeg/FFprobe khi cần decode video, materialize clip, hoặc audio preprocessing
- PostgreSQL và Elasticsearch chỉ cần cho backend tích hợp; không cần khi chạy notebook offline

Dependency được khóa trong `pyproject.toml` và `uv.lock`. Nhóm dependency mặc định gồm `pipeline` và `dev`; PyTorch/Paddle GPU dùng CUDA 12.6. TensorFlow Open Images được tách môi trường vì xung đột CUDA với PyTorch/Paddle.

## Cài đặt

```bash
git clone <repository-url>
cd 4US-1F-AIC-HCMC
uv sync
```

Chỉ cài dependency runtime, bỏ test:

```bash
uv sync --no-dev
```

Kiểm tra nhanh môi trường GPU trước khi chạy stage phụ thuộc CUDA:

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
ffmpeg -version
```

Nếu chỉ sử dụng detector TensorFlow Open Images, tạo venv tách biệt:

```bash
UV_PROJECT_ENVIRONMENT=.venv-openimages \
  uv sync --no-default-groups --group openimages
```

Caption 4-bit là tùy chọn:

```bash
uv sync --group caption-4bit
```

## Dữ liệu và cấu hình

Đặt dữ liệu theo mặc định trong `data/` (có thể thay đổi trong `BackEnd/CONFIG.py`):

```text
data/
├── video/                         # MP4 nguồn
├── keyframes/                     # keyframe ảnh
├── map-keyframes/                 # mapping/keyframe metadata
├── media-info-aic25-b1/media-info/# metadata video BTC
├── objects-aic25-b1/objects/      # object source BTC
├── objects-aic25-b1-jsonl/objects/# JSONL đã chuẩn hóa
└── models/
    ├── transnetv2-pytorch-weights.pth
    └── yolo26n.pt
```

Không commit video, dataset, weight, FAISS index hoặc secret. Sao chép mẫu biến môi trường nếu chạy backend có PostgreSQL:

```bash
cp .env.example .env
```

Sau đó đặt `DATABASE_URL` trong `.env`. Không đưa credential vào notebook, log hoặc commit. Một số notebook/stage có thêm cấu hình R2 hay FPT API; chỉ cung cấp bằng Secret của Kaggle hoặc biến môi trường cho lần chạy đó.

## Chạy backend tích hợp cục bộ

Backend tích hợp đọc danh sách video từ PostgreSQL, chạy các stage theo dependency, rồi ghi kết quả qua adapter database/FAISS:

```bash
uv run python -m BackEnd.app.pipeline.main_pipeline
```

Thứ tự chính là: shot → (keyframe, clip, OCR, tracking) → embedding frame/clip/shot. `PIPELINE_PARALLEL_MODE` trong cấu hình quyết định chạy `sequential`, `auto` hay `parallel`; chỉ dùng song song khi GPU preflight cho phép hai worker PyTorch/Paddle cùng tồn tại. Khi một GPU không đủ VRAM, chọn `sequential` để có kết quả ổn định hơn.

Một số entry point hữu ích:

```bash
# Demo keyframe trên một video BTC
uv run python scripts/run_extract_keyframe_demo.py --help

# Object detection YOLO trên thư mục keyframe
uv run python -m BackEnd.app.object_detection.run_detection --help

# Chuyển object JSONL của BTC thành SQL
uv run python -m BackEnd.app.object_detection.convert_aic25_objects_to_jsonl --help

# Tiền xử lý audio theo video/shot
uv run python -m BackEnd.app.audio_pre.run_preprocessing --help

# Benchmark artifact embedding
uv run python -m BackEnd.app.embedding.scripts.benchmark_embeddings --help
```

Hãy dùng `--help` để xem chính xác input/output của từng CLI trước khi chạy trên dữ liệu thật. Chi tiết nghiệp vụ từng module có trong các README dưới `BackEnd/app/`.

## Workflow Kaggle offline khuyến nghị

Mỗi notebook độc lập phải nhận một Kaggle Dataset đầu vào và chỉ ghi `/kaggle/working`. Đầu ra cần tải về và kiểm tra trước khi import.

| Notebook | Input tối thiểu | GPU | Artifact xuất |
| --- | --- | --- | --- |
| Shot Extractor | `videos.csv`, MP4, weight TransNetV2 | Có | `shot.csv`, `insert_shots.sql` |
| Keyframe Extractor | `videos.csv`, `shot.csv`, `keyframe.csv`, MP4 | Có | frame/keyframe CSV, ảnh, `insert_keyframes.sql` |
| Clip Extractor | `shot.csv`; MP4 chỉ khi xuất file clip | Không | `clipwindow.csv`, `insert_clipwindows.sql` |
| Object Detection | JSONL object BTC | Không | `insert_object_detection.sql` |
| Tracking | `videos.csv`, `shot.csv`, MP4, YOLO weight | Có | `insert_tracking.sql` |
| OCR | `shot.csv`, `keyframe.csv`, ảnh keyframe | Có | `insert_ocr.sql` |
| Frame Embedding | `keyframe.csv`, ảnh keyframe | Có | FAISS artifact, `insert_frame_embedding_records.sql` |
| Clip Embedding | `videos.csv`, `shot.csv`, `clipwindow.csv`, MP4 | Có | FAISS artifact, `insert_clip_embedding_records.sql` |
| Shot Embedding | `shot.csv`, Clip Embedding artifact cùng run/model | Không | FAISS artifact, `insert_shot_embedding_records.sql` |
| Caption | `videos.txt`, `shots.csv`, FPT Secret | Tùy chọn | `captions.sql` |

Các notebook CPU có thể chạy song song gồm Clip Extractor ở chế độ metadata-only, Object Detection ingestion và Shot Embedding. Sau khi có `shot.csv`, Keyframe Extractor, Clip Extractor và Tracking có thể được lên lịch độc lập; với một GPU nên chạy lần lượt để tránh OOM.

Notebook runbook đầy đủ: [Notebook/notebook_runbook.md](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/Notebook/notebook_runbook.md).

## Thứ tự import SQL

1. Nạp trước `classid`, `video` và các `frame` chính thức của BTC.
2. Nạp `shot`, sau đó `frame` bổ sung và `clipwindow`.
3. Nạp Object Detection, OCR và Tracking; Tracking yêu cầu `shot` và `classid`.
4. Nạp mapping embedding cùng đúng FAISS artifact của run/model/revision tương ứng.

`objectdetection`, `objecttrack` và `trackobservation` không có khóa idempotency đầy đủ cho toàn payload. Vì vậy chỉ import một SQL output một lần. Nếu run lỗi, giữ artifact để truy vết, tạo run ID mới và để DBA quyết định phần cần nạp; không dùng `DELETE`, `TRUNCATE` hay ghi đè production artifact để resume.

## Kiểm thử và kiểm tra trước khi chạy lớn

```bash
# Test code backend
uv run pytest -q

# Kiểm tra contract tĩnh cho toàn bộ notebook
uv run python Notebook/verify_notebook_contracts.py
```

Khuyến nghị chạy canary bằng lát cắt nhỏ (`VIDEO_START`/`VIDEO_END`) trước. Kiểm tra `summary.json`, `failures.jsonl`, CSV, số lượng record, SQL và artifact FAISS trước khi mở rộng toàn bộ dataset. Chỉ benchmark GPU sau khi xác nhận CUDA thực sự khả dụng trong runtime Python.

## Cấu trúc repo

```text
BackEnd/
├── CONFIG.py                 # cấu hình tập trung
├── app/
│   ├── pipeline/             # orchestration và ingestion
│   ├── contracts/            # schema dùng chung
│   ├── database/             # PostgreSQL, Elasticsearch, FAISS
│   ├── embedding/            # encoding, artifact, FAISS
│   ├── shot_extractor/
│   ├── keyframe_extractor/
│   ├── clip_extractor/
│   ├── ocr/
│   ├── object_detection/
│   ├── tracking/
│   ├── audio_pre/
│   └── ASR/
├── tests/
Notebook/                     # notebook Kaggle + runbook
scripts/                      # demo/tải dữ liệu hỗ trợ
```

## Tài liệu liên quan

- [Runbook notebook](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/Notebook/notebook_runbook.md)
- [Checklist notebook](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/Notebook/check_list_for_notebook.md)
- [Đề xuất cải thiện pipeline](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/proposal.md)
- [Tài liệu clip extractor](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/BackEnd/app/clip_extractor/README.md)
- [Tài liệu OCR](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/BackEnd/app/ocr/README.md)
- [Tài liệu tracking](/home/nhminh/AI_Project/4US-1F-AIC-HCMC/BackEnd/app/tracking/README.md)
