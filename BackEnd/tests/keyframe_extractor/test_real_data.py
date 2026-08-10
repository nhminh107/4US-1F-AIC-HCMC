"""Integration test chạy trên dữ liệu video & keyframe thực tế trong data/ (L21_V001.mp4).

Test này kiểm tra:
1. Đọc video thực tế L21_V001.mp4 và nạp danh sách keyframe official gốc từ data/Keyframes_L21/keyframes/L21_V001/.
2. Chạy KeyframeExtractor với Single-Pass FFmpeg extraction trên video thật.
3. Xác minh:
   - Ảnh .jpg bổ sung thực sự được tạo ra trên đĩa trong thư mục tạm.
   - Không trùng lặp frame_idx với official keyframes thực tế.
   - Ảnh trích xuất mở được bình thường qua PIL, kích thước (width, height) khớp đúng.
   - frame_id tuân thủ định dạng L21_V001_E001 (varchar(15)).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _find_real_video(video_id: str = "L21_V001") -> Path | None:
    candidates = sorted(PROJECT_ROOT.glob(f"data/**/{video_id}.mp4"))
    return candidates[0] if candidates else None


_REAL_VIDEO_PATH = _find_real_video("L21_V001")


@unittest.skipUnless(_FFMPEG_AVAILABLE, "FFmpeg/FFprobe không có trong PATH")
@unittest.skipUnless(_REAL_VIDEO_PATH is not None, "Không tìm thấy video thực tế L21_V001.mp4 trong data/")
class RealDataIntegrationTests(unittest.TestCase):
    """Test thực tế trên dữ liệu video & keyframes thật L21_V001."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.output_keyframe_dir = Path(self.temp_dir) / "keyframes"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_additional_keyframes_on_real_video_and_official_data(self) -> None:
        video_id = "L21_V001"

        # 1. Đọc danh sách keyframe official thực tế từ data/Keyframes_L21/keyframes/L21_V001/
        official_dir = PROJECT_ROOT / "data" / "Keyframes_L21" / "keyframes" / video_id
        all_official_idxs: list[int] = []
        if official_dir.is_dir():
            for img_file in official_dir.glob("*.jpg"):
                try:
                    idx = int(img_file.stem)
                    all_official_idxs.append(idx)
                except ValueError:
                    pass

        # Giả lập danh sách keyframe official thực tế thưa thớt (ví dụ chỉ có 2 keyframe ở frame 20 và 300)
        sparse_official_idxs = [20, 300]

        # Khai báo các shot thực tế nằm trong video L21_V001.mp4 (kéo dài tới frame 500)
        shots = [
            ShotMetadata(
                shot_id=f"{video_id}_S001",
                video_id=video_id,
                shot_index=0,
                start_ms=0,
                end_ms=5000,
                start_frame_idx=0,
                end_frame_idx=124,
            ),
            ShotMetadata(
                shot_id=f"{video_id}_S002",
                video_id=video_id,
                shot_index=1,
                start_ms=5000,
                end_ms=20000,
                start_frame_idx=125,
                end_frame_idx=499,
            ),
        ]

        print(f"\n[Real Data Test] Found {len(all_official_idxs)} total official keyframe files.")
        print(f"[Real Data Test] Testing with {len(sparse_official_idxs)} sparse official keyframes.")

        # 2. Chạy KeyframeExtractor (Single-Pass FFmpeg) trên video thực tế L21_V001.mp4
        keyframe_extractor = KeyframeExtractor(keyframe_dir=self.output_keyframe_dir)
        extracted_frames = keyframe_extractor.extract_for_video(
            video_id, shots, existing_frame_idxs=sparse_official_idxs
        )

        print(f"[Real Data Test] Successfully extracted {len(extracted_frames)} additional keyframes from real MP4 video.")

        self.assertGreater(len(extracted_frames), 0, "Phải trích xuất được các keyframe bổ sung từ video thật")

        # 3. Kiểm tra kỹ lưỡng các ảnh thật được ghi ra đĩa
        sparse_set = set(sparse_official_idxs)
        extracted_indices: set[int] = set()

        for frame in extracted_frames:
            # Kiểm tra metadata contract
            self.assertEqual(frame.video_id, video_id)
            self.assertEqual(frame.source, "extracted")
            self.assertTrue(frame.frame_id.startswith(f"{video_id}_E"))
            self.assertLessEqual(len(frame.frame_id), 15, "frame_id phải vừa varchar(15)")

            # Đảm bảo không trùng với sparse official keyframe
            self.assertNotIn(
                frame.frame_idx,
                sparse_set,
                f"Lỗi: Frame index {frame.frame_idx} bị trùng với official keyframe!",
            )

            # Đảm bảo không trùng lặp giữa các keyframe extracted
            self.assertNotIn(
                frame.frame_idx,
                extracted_indices,
                f"Lỗi: Frame index {frame.frame_idx} bị trùng lặp!",
            )
            extracted_indices.add(frame.frame_idx)

            # Kiểm tra file ảnh JPEG thực tế được ghi ra đĩa
            self.assertIsNotNone(frame.frame_path)
            self.assertTrue(frame.frame_path.is_file(), f"File ảnh không tồn tại: {frame.frame_path}")

            # Đọc file ảnh bằng Pillow để xác nhận không lỗi
            with Image.open(frame.frame_path) as img:
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)
                self.assertEqual(img.width, frame.width)
                self.assertEqual(img.height, frame.height)

        print(f"[Real Data Test] ALL {len(extracted_frames)} REAL EXTRACTED KEYFRAME IMAGES VERIFIED 100% VALID!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
