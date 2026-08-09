"""Decode và lưu ảnh keyframe bằng FFmpeg.

Module này chịu trách nhiệm:
- Sử dụng FFmpeg CLI để cắt chính xác các frame theo index (`frame_idx`) từ video `.mp4`.
- Hỗ trợ Single-Pass extraction (1 lượt đọc duy nhất cho nhiều frame) giúp tối ưu hiệu năng.
- Ghi ảnh trực tiếp ra đĩa dưới dạng JPEG tại đường dẫn được chỉ định.
- Trả về thông tin `(width, height)` của từng ảnh đã trích xuất.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from collections.abc import Sequence
from pathlib import Path
from PIL import Image

_MISSING_FFMPEG_HINT = (
    "'{executable}' executable not found on PATH. Install FFmpeg and make sure it is on PATH."
)


def extract_and_save_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    output_paths: Sequence[Path],
) -> list[tuple[int, int]]:
    """Trích xuất và lưu các frame theo `frame_indices` từ video thành ảnh JPEG.

    Args:
        video_path: Đường dẫn tới file video `.mp4`.
        frame_indices: Danh sách các `frame_idx` (0-based) cần trích xuất.
        output_paths: Danh sách các đường dẫn file output `.jpg` tương ứng.

    Returns:
        Danh sách các cặp `(width, height)` tương ứng với từng file ảnh đã trích xuất.
    """

    if len(frame_indices) != len(output_paths):
        raise ValueError(
            f"Mismatched lengths: {len(frame_indices)} frame_indices vs {len(output_paths)} output_paths"
        )

    if not frame_indices:
        return []

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at '{video_path}'")

    # Đảm bảo các thư mục cha của output_paths tồn tại
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Nếu chỉ có 1 frame index
    if len(frame_indices) == 1:
        _extract_single_frame(video_path, frame_indices[0], output_paths[0])
    else:
        # Thử Single-Pass extraction trước để tối ưu hiệu năng; fallback nếu có sự cố
        try:
            _extract_multiple_frames_single_pass(video_path, frame_indices, output_paths)
        except Exception:
            _extract_multiple_frames_fallback(video_path, frame_indices, output_paths)

    # Đọc width, height của từng file ảnh vừa tạo
    dimensions: list[tuple[int, int]] = []
    for path in output_paths:
        if not path.is_file():
            raise RuntimeError(f"Failed to extract frame: output image not created at '{path}'")
        with Image.open(path) as img:
            dimensions.append((img.width, img.height))

    return dimensions


def _extract_single_frame(video_path: Path, frame_idx: int, output_path: Path) -> None:
    """Cắt 1 frame duy nhất theo `frame_idx` bằng FFmpeg select filter."""

    filter_str = f"select=eq(n\\,{frame_idx})"
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(video_path),
        "-vf", filter_str,
        "-vframes", "1",
        str(output_path),
    ]

    try:
        subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(_MISSING_FFMPEG_HINT.format(executable="ffmpeg")) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else ""
        raise RuntimeError(
            f"ffmpeg failed to extract frame_idx {frame_idx} from '{video_path}': {stderr.strip()}"
        ) from error


def _extract_multiple_frames_single_pass(
    video_path: Path, frame_indices: Sequence[int], output_paths: Sequence[Path]
) -> None:
    """Cắt nhiều frame trong 1 lượt pass duy nhất bằng FFmpeg với select filter và vsync vfr."""

    sorted_pairs = sorted(zip(frame_indices, output_paths), key=lambda x: x[0])
    sorted_indices = [p[0] for p in sorted_pairs]
    sorted_outputs = [p[1] for p in sorted_pairs]

    select_conditions = "+".join(f"eq(n\\,{idx})" for idx in sorted_indices)
    filter_str = f"select={select_conditions}"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pattern = Path(temp_dir) / "frame_%04d.jpg"
        command = [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(video_path),
            "-vf", filter_str,
            "-vsync", "vfr",
            str(temp_pattern),
        ]

        subprocess.run(command, capture_output=True, check=True)

        extracted_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if len(extracted_files) != len(sorted_outputs):
            raise RuntimeError(
                f"Expected {len(sorted_outputs)} frames extracted in single-pass, got {len(extracted_files)}"
            )

        for src, dest in zip(extracted_files, sorted_outputs):
            shutil.move(str(src), str(dest))


def _extract_multiple_frames_fallback(
    video_path: Path, frame_indices: Sequence[int], output_paths: Sequence[Path]
) -> None:
    """Fallback trích xuất tuần tự từng frame nếu single-pass gặp sự cố."""

    for idx, path in zip(frame_indices, output_paths):
        _extract_single_frame(video_path, idx, path)
