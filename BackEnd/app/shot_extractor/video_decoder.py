"""Decode video bằng FFmpeg, phục vụ phát hiện ranh giới shot.

Quyết định thiết kế (xem ``Markdown_Doc/module_shot_keyframe.md`` mục 2.3):
decode bằng công cụ dòng lệnh ``ffmpeg``/``ffprobe``, không dùng OpenCV, để
nhất quán với các module khác của nhóm và tránh thêm một dependency nặng.

Module này chỉ có đúng hai trách nhiệm:

- ``probe_fps``: đọc frame rate của video từ metadata của container.
- ``decode_frames_for_transnet``: decode toàn bộ frame, downscale về đúng
  kích thước RGB 48x27 mà TransNetV2 yêu cầu, gộp thành một mảng ``uint8``
  duy nhất trong bộ nhớ.

Cả hai đều raise ``RuntimeError`` kèm thông báo có thể hành động được (thiếu
executable, file hỏng, stream rỗng) thay vì âm thầm fallback, đúng theo quy
tắc của dự án là không được nuốt lỗi trong một pipeline chạy batch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from BackEnd.app.shot_extractor.transnetv2_model import INPUT_HEIGHT, INPUT_WIDTH

_FRAME_BYTES = INPUT_HEIGHT * INPUT_WIDTH * 3  # rgb24, 3 byte mỗi pixel

_MISSING_FFMPEG_HINT = (
    "'{executable}' executable not found on PATH. Install FFmpeg (it "
    "bundles both 'ffmpeg' and 'ffprobe') and make sure it is on PATH. "
    "See Markdown_Doc/module_shot_keyframe.md section 2.3 and "
    "Markdown_Doc/shot_extractor_pipeline.md for setup notes."
)


def probe_fps(video_path: Path) -> float:
    """Trả về frame rate (frames/giây) của video stream.

    Ưu tiên ``avg_frame_rate`` (frame rate trung bình thật của stream, ổn
    định với nguồn có frame rate biến đổi) và fallback sang ``r_frame_rate``
    (frame rate danh nghĩa) khi giá trị trung bình không khả dụng (ví dụ
    ``0/0`` mà ffprobe trả về với một số container).
    """

    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(_MISSING_FFMPEG_HINT.format(executable="ffprobe")) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffprobe failed to read '{video_path}': {error.stderr.strip()}"
        ) from error

    fields = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    fps = _parse_frame_rate(fields.get("avg_frame_rate", "")) or _parse_frame_rate(
        fields.get("r_frame_rate", "")
    )
    if fps is None:
        raise RuntimeError(
            f"Could not determine a valid frame rate for '{video_path}' "
            f"(ffprobe reported avg_frame_rate={fields.get('avg_frame_rate')!r}, "
            f"r_frame_rate={fields.get('r_frame_rate')!r})."
        )
    return fps


def decode_frames_for_transnet(video_path: Path) -> np.ndarray:
    """Decode toàn bộ frame của ``video_path`` về đúng kích thước RGB 48x27 của TransNetV2.

    Trả về mảng ``uint8`` shape ``[frame_count, 27, 48, 3]``, theo đúng thứ
    tự hiển thị/decode, bắt đầu từ ``frame_idx = 0``. Toàn bộ video được nạp
    hết vào bộ nhớ: ở mức 3.888 byte/frame thì kể cả video dài cũng chỉ tốn
    vài chục MB, không đáng kể so với file MP4 gốc.
    """

    command = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale={INPUT_WIDTH}:{INPUT_HEIGHT}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(_MISSING_FFMPEG_HINT.format(executable="ffmpeg")) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else ""
        raise RuntimeError(f"ffmpeg failed to decode '{video_path}': {stderr.strip()}") from error

    raw = result.stdout
    if len(raw) % _FRAME_BYTES != 0:
        raise RuntimeError(
            f"Decoded byte stream for '{video_path}' is not a whole number of "
            f"frames: {len(raw)} bytes is not divisible by {_FRAME_BYTES} "
            f"({INPUT_HEIGHT}x{INPUT_WIDTH}x3)."
        )

    frame_count = len(raw) // _FRAME_BYTES
    if frame_count == 0:
        raise RuntimeError(f"Decoded 0 frames from '{video_path}'; the file may be empty or corrupted.")

    return np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, INPUT_HEIGHT, INPUT_WIDTH, 3)


def _parse_frame_rate(raw_fraction: str) -> float | None:
    """Parse chuỗi frame-rate dạng ``"num/den"`` của ffprobe thành FPS kiểu float."""

    numerator_str, _, denominator_str = raw_fraction.partition("/")
    try:
        numerator = float(numerator_str)
        denominator = float(denominator_str) if denominator_str else 1.0
    except ValueError:
        return None
    if denominator == 0:
        return None
    fps = numerator / denominator
    return fps if fps > 0 else None
