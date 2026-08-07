"""Test xác nhận ShotExtractor.extract() nối đúng decode -> model -> shot_boundary,
mà không cần FFmpeg, trọng số TransNetV2 thật, hay GPU.

Việc decode bằng FFmpeg và trọng số pretrain được mock/thay thế; chỉ phần
orchestration và phép toán sliding-window inference chạy thật (trên một
model vừa khởi tạo, chưa train — prediction của nó vô nghĩa, nhưng shape
tensor và luồng điều khiển thì đúng y hệt bản production dùng).
``test_shot_extractor_smoke.py`` (có skip-guard) mới là nơi kiểm tra đường
đi end-to-end thật trên máy đã cấu hình đầy đủ.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from BackEnd.app.shot_extractor.shot_extractor import ShotExtractor, _iter_windows
from BackEnd.app.shot_extractor.transnetv2_model import TransNetV2


class IterWindowsTests(unittest.TestCase):
    """Kiểm tra phần đệm của sliding-window phủ đúng mọi frame đầu vào ít nhất một lần."""

    def test_window_shape_and_coverage_for_various_frame_counts(self) -> None:
        for frame_count in (1, 49, 50, 51, 100, 137, 250):
            with self.subTest(frame_count=frame_count):
                frames = np.zeros((frame_count, 27, 48, 3), dtype=np.uint8)
                windows = list(_iter_windows(frames))
                for window in windows:
                    self.assertEqual(window.shape, (100, 27, 48, 3))
                # Mỗi cửa sổ đóng góp 50 frame giữa của nó; phải có đủ cửa sổ
                # để phủ được mọi frame thật ít nhất một lần.
                self.assertGreaterEqual(len(windows) * 50, frame_count)


class ShotExtractorWiringTests(unittest.TestCase):
    """Kiểm tra extract() chạy end-to-end với việc decode FFmpeg/nạp weights được thay thế."""

    def _build_extractor_with_untrained_model(self) -> ShotExtractor:
        extractor = ShotExtractor(video_dir="data/video", window_batch_size=4)
        extractor._model = TransNetV2().to(extractor.device)
        return extractor

    def test_extract_returns_valid_shots_using_mocked_decode(self) -> None:
        extractor = self._build_extractor_with_untrained_model()
        fake_frames = np.random.randint(0, 255, size=(300, 27, 48, 3), dtype=np.uint8)

        with (
            patch(
                "BackEnd.app.shot_extractor.shot_extractor.probe_fps", return_value=25.0
            ),
            patch(
                "BackEnd.app.shot_extractor.shot_extractor.decode_frames_for_transnet",
                return_value=fake_frames,
            ),
            patch.object(ShotExtractor, "_resolve_video_path", return_value="fake.mp4"),
        ):
            shots = extractor.extract("L21_V001")

        self.assertGreater(len(shots), 0)
        previous_end_frame_idx = -1
        for index, shot in enumerate(shots):
            self.assertEqual(shot.shot_index, index)
            self.assertEqual(shot.video_id, "L21_V001")
            self.assertGreater(shot.end_ms, shot.start_ms)
            self.assertGreater(shot.end_frame_idx, previous_end_frame_idx)
            previous_end_frame_idx = shot.end_frame_idx
        # Mọi frame nguồn phải thuộc về (nhiều nhất) một shot, giới hạn đúng
        # theo frame_idx cuối cùng đã decode.
        self.assertLessEqual(shots[-1].end_frame_idx, 299)

    def test_extract_raises_when_video_file_is_missing(self) -> None:
        extractor = self._build_extractor_with_untrained_model()
        with self.assertRaises(FileNotFoundError):
            extractor.extract("does_not_exist_video_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
