"""Unit test cho logic thuần của shot-boundary (không cần FFmpeg/PyTorch/model).

Đây là các test ở mức nhỏ nhất, có ý nghĩa nhất cho module 2.3 (xem
``agent.md`` mục 12): các hàm thuần thao tác trên mảng NumPy và data
contract, chạy trước tiên và nhanh nhất vì không đụng tới video thật, trọng
số model, hay GPU.
"""

from __future__ import annotations

import unittest

import numpy as np

from BackEnd.app.shot_extractor.shot_boundary import (
    predictions_to_scene_frames,
    scene_frames_to_shots,
)


class PredictionsToSceneFramesTests(unittest.TestCase):
    """Kiểm tra ngữ nghĩa ranh giới của hàm `predictions_to_scenes` đã port."""

    def test_single_transition_splits_into_two_shots(self) -> None:
        predictions = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
        scenes = predictions_to_scene_frames(predictions, threshold=0.5)
        self.assertEqual(scenes, [(0, 3), (5, 7)])

    def test_multi_frame_transition_excludes_interior_frames(self) -> None:
        # Frame 3-6 đều "đang chuyển cảnh"; chỉ frame 3 kết thúc shot 0 và
        # chỉ frame 7 (frame nội dung đầu tiên sau đó) bắt đầu shot 1.
        # Frame 4-6 không thuộc shot nào cả, đúng theo hành vi gốc của TransNetV2.
        predictions = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float32)
        scenes = predictions_to_scene_frames(predictions, threshold=0.5)
        self.assertEqual(scenes, [(0, 3), (7, 8)])

    def test_no_transitions_returns_single_shot(self) -> None:
        predictions = np.zeros(10, dtype=np.float32)
        scenes = predictions_to_scene_frames(predictions, threshold=0.5)
        self.assertEqual(scenes, [(0, 9)])

    def test_all_frames_above_threshold_returns_single_shot_fallback(self) -> None:
        predictions = np.ones(10, dtype=np.float32)
        scenes = predictions_to_scene_frames(predictions, threshold=0.5)
        self.assertEqual(scenes, [(0, 9)])

    def test_threshold_is_exclusive(self) -> None:
        # Điểm số đúng bằng threshold KHÔNG được tính là ranh giới.
        predictions = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        scenes = predictions_to_scene_frames(predictions, threshold=0.5)
        self.assertEqual(scenes, [(0, 2)])

    def test_empty_predictions_raises(self) -> None:
        with self.assertRaises(ValueError):
            predictions_to_scene_frames(np.array([], dtype=np.float32))

    def test_non_1d_predictions_raises(self) -> None:
        with self.assertRaises(ValueError):
            predictions_to_scene_frames(np.zeros((2, 2), dtype=np.float32))


class SceneFramesToShotsTests(unittest.TestCase):
    """Kiểm tra quy đổi ms, gộp shot ngắn, và việc dựng ShotMetadata."""

    FPS = 25.0  # 40ms/frame, chọn giá trị này để phép tính ms ra số chẵn, dễ assert

    def test_single_scene_produces_one_shot_with_expected_ids_and_bounds(self) -> None:
        shots = scene_frames_to_shots(
            [(0, 249)], video_id="L21_V001", fps=self.FPS, min_shot_duration_ms=500
        )
        self.assertEqual(len(shots), 1)
        shot = shots[0]
        self.assertEqual(shot.shot_id, "L21_V001_S000")
        self.assertEqual(shot.video_id, "L21_V001")
        self.assertEqual(shot.shot_index, 0)
        self.assertEqual(shot.start_frame_idx, 0)
        self.assertEqual(shot.end_frame_idx, 249)
        self.assertEqual(shot.start_ms, 0)
        self.assertEqual(shot.end_ms, 10_000)  # 250 frame / 25 fps

    def test_two_long_scenes_stay_separate_and_renumbered(self) -> None:
        # 125 frame (5000ms) rồi 175 frame (7000ms): cả hai đều vượt xa 500ms.
        shots = scene_frames_to_shots(
            [(0, 124), (125, 299)],
            video_id="L21_V001",
            fps=self.FPS,
            min_shot_duration_ms=500,
        )
        self.assertEqual([s.shot_index for s in shots], [0, 1])
        self.assertEqual([s.shot_id for s in shots], ["L21_V001_S000", "L21_V001_S001"])
        self.assertEqual(shots[0].end_ms, shots[1].start_ms)  # bàn giao liền mạch, không hở

    def test_short_shot_merges_into_preceding_shot(self) -> None:
        # Shot 0: 200 frame = 8000ms (dài). Shot 1: 5 frame = 200ms (ngắn, gộp ngược).
        shots = scene_frames_to_shots(
            [(0, 199), (200, 204)],
            video_id="L21_V001",
            fps=self.FPS,
            min_shot_duration_ms=500,
        )
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame_idx, 0)
        self.assertEqual(shots[0].end_frame_idx, 204)

    def test_short_first_shot_with_no_predecessor_merges_forward(self) -> None:
        # Shot 0: 5 frame = 200ms (ngắn, là shot đầu -> không có gì để gộp ngược).
        # Shot 1: 45 frame = 1800ms (dài).
        shots = scene_frames_to_shots(
            [(0, 4), (5, 49)],
            video_id="L21_V001",
            fps=self.FPS,
            min_shot_duration_ms=500,
        )
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame_idx, 0)
        self.assertEqual(shots[0].end_frame_idx, 49)

    def test_cascading_short_shots_all_absorb_into_one_neighbor(self) -> None:
        # Ba shot rất ngắn liên tiếp (80ms mỗi shot, tổng 240ms) vẫn dưới mốc
        # 500ms, nên sau khi gộp ngược thành một nhóm 240ms thì vẫn không đủ
        # để đứng riêng, phải tiếp tục gộp xuôi vào shot dài theo sau, gộp
        # tất cả thành một shot duy nhất.
        shots = scene_frames_to_shots(
            [(0, 1), (2, 3), (4, 5), (6, 99)],
            video_id="L21_V001",
            fps=self.FPS,
            min_shot_duration_ms=500,
        )
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame_idx, 0)
        self.assertEqual(shots[0].end_frame_idx, 99)

    def test_cascading_short_shots_that_clear_threshold_stay_their_own_shot(self) -> None:
        # Tương tự nhưng mỗi shot ngắn đủ dài (200ms) để nhóm sau khi gộp
        # ngược đã tự vượt 500ms: nó KHÔNG được tiếp tục nuốt shot dài phía sau.
        shots = scene_frames_to_shots(
            [(0, 4), (5, 9), (10, 14), (15, 99)],
            video_id="L21_V001",
            fps=self.FPS,
            min_shot_duration_ms=500,
        )
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0].start_frame_idx, 0)
        self.assertEqual(shots[0].end_frame_idx, 14)
        self.assertEqual(shots[1].start_frame_idx, 15)
        self.assertEqual(shots[1].end_frame_idx, 99)

    def test_entire_short_video_collapses_to_single_shot(self) -> None:
        # Một video mà mọi scene phát hiện được đều ngắn: sau khi gộp hết
        # vào nhau thì không còn gì khác để gộp tiếp, nên giữ nguyên một shot.
        shots = scene_frames_to_shots(
            [(0, 4), (5, 9)], video_id="L21_V001", fps=self.FPS, min_shot_duration_ms=500
        )
        self.assertEqual(len(shots), 1)

    def test_non_positive_fps_raises(self) -> None:
        with self.assertRaises(ValueError):
            scene_frames_to_shots([(0, 9)], video_id="L21_V001", fps=0)

    def test_empty_scene_frames_raises(self) -> None:
        with self.assertRaises(ValueError):
            scene_frames_to_shots([], video_id="L21_V001", fps=self.FPS)

    def test_out_of_order_scene_frames_raise(self) -> None:
        with self.assertRaises(ValueError):
            scene_frames_to_shots(
                [(10, 20), (0, 9)], video_id="L21_V001", fps=self.FPS
            )

    def test_overlapping_scene_frames_raise(self) -> None:
        with self.assertRaises(ValueError):
            scene_frames_to_shots(
                [(0, 20), (10, 30)], video_id="L21_V001", fps=self.FPS
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
