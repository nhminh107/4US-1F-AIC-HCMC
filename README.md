# AIC HCMC video preprocessing

Dependency của project được quản lý tập trung bằng `pyproject.toml` và
`uv.lock`. Project yêu cầu Ubuntu x86_64, Python 3.12 và CUDA 12.6 cho pipeline
GPU chính.

## Cài môi trường pipeline

```bash
uv sync
```

Lệnh trên cài hai group mặc định `pipeline` và `dev`. Trên máy chỉ chạy
preprocessing, có thể bỏ dependency test:

```bash
uv sync --no-dev
```

Chạy command Python qua môi trường đã lock:

```bash
uv run python -m BackEnd.app.pipeline.main_pipeline
uv run pytest -q
```

PaddlePaddle GPU và PyTorch dùng chung CUDA runtime do PyTorch cung cấp. Cấu
hình metadata tương thích nằm trong `pyproject.toml`; không cần cài Paddle bằng
`--no-deps` thủ công.

## TensorFlow Open Images

Faster R-CNN TensorFlow không thuộc main pipeline và phải ở môi trường riêng do
CUDA dependency xung đột với PyTorch/Paddle:

```bash
UV_PROJECT_ENVIRONMENT=.venv-openimages \
  uv sync --no-default-groups --group openimages
```

## Caption 4-bit tùy chọn

```bash
uv sync --group caption-4bit
```

Không cần thư mục `.uv`; uv dùng `pyproject.toml`, `uv.lock` và `.venv/`.
