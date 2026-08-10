"""Trích xuất ranh giới shot: một video -> danh sách shot có thứ tự, không chồng lấn.

Cài đặt module 2.3 của ``Markdown_Doc/module_shot_keyframe.md``, dùng
TransNetV2 (bản port PyTorch, xem ``transnetv2_model.py``) để phát hiện ranh
giới shot. Xem ``Markdown_Doc/shot_extractor_pipeline.md`` để có bài viết
đầy đủ về pipeline, thuật toán sliding-window, và các vấn đề thực tế gặp
phải khi xây dựng module này.

``ShotExtractor.extract(video_id)`` là entry point công khai duy nhất, và
chữ ký của nó là một contract cố định dùng chung với phần còn lại của
pipeline (``app/pipeline/`` gọi hàm này, ``app/keyframe_extractor/`` tiêu
thụ kết quả trả về) — không được đổi chữ ký nếu không cập nhật mọi nơi gọi.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from BackEnd.CONFIG import (
    PROJECT_ROOT,
    SHOT_VIDEO_DIR as DEFAULT_VIDEO_DIR,
    SHOT_WEIGHTS_PATH as DEFAULT_WEIGHTS_PATH,
    SHOT_WINDOW_BATCH_SIZE,
)
from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.shot_extractor.shot_boundary import (
    DEFAULT_MIN_SHOT_DURATION_MS,
    DEFAULT_THRESHOLD,
    predictions_to_scene_frames,
    scene_frames_to_shots,
)
from BackEnd.app.shot_extractor.transnetv2_model import TransNetV2
from BackEnd.app.shot_extractor.video_decoder import decode_frames_for_transnet, probe_fps

# Cấu hình sliding-window inference cố định của TransNetV2 (xem
# https://github.com/soCzech/TransNetV2 inference/transnetv2.py `predict_frames`):
# mỗi cửa sổ 100 frame chỉ đóng góp prediction cho 50 frame ở giữa,
# với 25 frame ngữ cảnh mỗi bên được copy từ các cửa sổ lân cận.
_WINDOW_SIZE = 100
_WINDOW_HOP = 50
_WINDOW_CONTEXT = 25

_MISSING_WEIGHTS_MESSAGE = (
    "TransNetV2 weights not found at '{path}'. Convert the official "
    "TensorFlow checkpoint to PyTorch using the 'inference-pytorch/' "
    "convert_weights.py script from https://github.com/soCzech/TransNetV2 "
    "(see Markdown_Doc/module_shot_keyframe.md section 2.3), then place the "
    "resulting 'transnetv2-pytorch-weights.pth' at this path, or pass an "
    "explicit weights_path / set TRANSNETV2_WEIGHTS_PATH."
)


class ShotExtractor:
    """Trích xuất ranh giới shot cho một video của BTC bằng TransNetV2."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_shot_duration_ms: int = DEFAULT_MIN_SHOT_DURATION_MS,
        video_dir: str | Path = DEFAULT_VIDEO_DIR,
        window_batch_size: int = SHOT_WINDOW_BATCH_SIZE,
    ) -> None:
        """Cấu hình extractor. Model TransNetV2 được nạp lazy (khi thật sự cần).

        Args:
            weights_path: Đường dẫn tới checkpoint ``.pth`` đã convert. Mặc
                định là ``data/models/transnetv2-pytorch-weights.pth``.
            device: Chuỗi device của torch (vd ``"cuda"``, ``"cpu"``). Mặc
                định tự chọn CUDA nếu có, nếu không thì dùng CPU kèm cảnh báo
                tường minh — dự án yêu cầu chạy GPU (xem ``agent.md`` mục 3)
                nhưng module này vẫn phải chạy được để phát triển/test trên
                máy không có GPU.
            threshold: Xác suất sigmoid mà trên đó một frame được tính là
                ranh giới shot. Mặc định theo đúng giá trị mặc định của
                TransNetV2 (0.5).
            min_shot_duration_ms: Shot ngắn hơn giá trị này bị gộp vào shot
                lân cận vì coi là nhiễu của detector.
            video_dir: Thư mục chứa file ``<video_id>.mp4``.
            window_batch_size: Số cửa sổ inference 100-frame chạy trong mỗi
                lượt forward pass. Chỉ là tham số tối ưu throughput, không
                làm thay đổi kết quả.
        """

        self.weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
        self.device = _resolve_device(device)
        self.threshold = threshold
        self.min_shot_duration_ms = min_shot_duration_ms
        self.video_dir = Path(video_dir)
        self.window_batch_size = window_batch_size

        self._model: TransNetV2 | None = None

    def extract(self, video_id: str) -> list[ShotMetadata]:
        """Phát hiện ranh giới shot cho ``video_id`` và trả về danh sách shot có thứ tự.

        Resolve video nguồn từ ``data/video/{video_id}.mp4``, decode bằng
        FFmpeg, chạy TransNetV2 để chấm điểm từng frame, quy đổi điểm số
        thành ranh giới shot theo ``self.threshold``, và gộp các shot ngắn
        hơn ``self.min_shot_duration_ms``.
        """

        video_path = self._resolve_video_path(video_id)
        fps = probe_fps(video_path)
        frames = decode_frames_for_transnet(video_path)

        predictions = self._predict(frames)
        scene_frames = predictions_to_scene_frames(predictions, threshold=self.threshold)

        return scene_frames_to_shots(
            scene_frames,
            video_id=video_id,
            fps=fps,
            min_shot_duration_ms=self.min_shot_duration_ms,
        )

    def _resolve_video_path(self, video_id: str) -> Path:
        video_path = self.video_dir / f"{video_id}.mp4"
        if not video_path.is_file():
            candidates = sorted(PROJECT_ROOT.glob(f"data/**/{video_id}.mp4"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"Video not found for '{video_id}': {video_path}")
        return video_path

    def _get_model(self) -> TransNetV2:
        """Nạp và cache model TransNetV2 khi dùng lần đầu (một lần cho mỗi instance)."""

        if self._model is not None:
            return self._model

        if not self.weights_path.is_file():
            raise FileNotFoundError(_MISSING_WEIGHTS_MESSAGE.format(path=self.weights_path))

        model = TransNetV2()
        state_dict = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        self._model = model
        return model

    def _predict(self, frames: np.ndarray) -> np.ndarray:
        """Chạy TransNetV2 inference kiểu sliding-window, trả về điểm số cho từng frame.

        Port lại đúng sơ đồ cửa sổ của bản gốc ``predict_frames``
        (window=100, hop=50, ngữ cảnh 25 frame mỗi bên, đệm bằng cách lặp
        lại frame ở hai đầu), nhưng gộp nhiều cửa sổ vào một batch mỗi lượt
        forward pass để tăng throughput trên GPU.
        """

        model = self._get_model()
        windows = list(_iter_windows(frames))

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch_start in range(0, len(windows), self.window_batch_size):
                batch = windows[batch_start : batch_start + self.window_batch_size]
                batch_tensor = torch.from_numpy(np.stack(batch)).to(self.device)

                logits, _ = model(batch_tensor)
                middle_probs = torch.sigmoid(logits)[:, _WINDOW_CONTEXT : _WINDOW_CONTEXT + _WINDOW_HOP, 0]
                chunks.append(middle_probs.cpu().numpy())

        predictions = np.concatenate(chunks, axis=0).reshape(-1)
        return predictions[: len(frames)]


def _iter_windows(frames: np.ndarray):
    """Sinh các cửa sổ kích thước cố định, đệm ở hai đầu, trên ``frames`` cho TransNetV2.

    Cửa sổ đầu/cuối được đệm bằng cách lặp lại frame thật đầu tiên/cuối cùng
    (không bao giờ dùng frame đen/toàn số 0), đúng như bản gốc, để các frame
    gần đầu/cuối video vẫn được chấm điểm dựa trên ngữ cảnh thật thay vì
    frame giả.
    """

    total_frames = len(frames)
    trailing_pad = _WINDOW_CONTEXT + _WINDOW_HOP - (
        total_frames % _WINDOW_HOP if total_frames % _WINDOW_HOP != 0 else _WINDOW_HOP
    )

    padded = np.concatenate(
        [
            np.repeat(frames[:1], _WINDOW_CONTEXT, axis=0),
            frames,
            np.repeat(frames[-1:], trailing_pad, axis=0),
        ],
        axis=0,
    )

    pointer = 0
    while pointer + _WINDOW_SIZE <= len(padded):
        yield padded[pointer : pointer + _WINDOW_SIZE]
        pointer += _WINDOW_HOP


def _resolve_device(device: str | None) -> torch.device:
    """Resolve device dùng để inference, luôn cảnh báo (không bao giờ âm thầm) khi fallback về CPU."""

    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}' but torch.cuda.is_available() is False."
            )
        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")

    warnings.warn(
        "CUDA is not available; ShotExtractor will run TransNetV2 on CPU, "
        "which is significantly slower than the GPU execution required by "
        "agent.md section 3. Pass device='cuda' explicitly to fail loudly "
        "instead of falling back.",
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.device("cpu")
