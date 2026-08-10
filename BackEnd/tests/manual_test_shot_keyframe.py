"""Manual Accuracy Test Script cho Module Extract Shot và Extract Keyframe.

Cung cấp CLI interface và visualization cho người dùng tự chạy kiểm thử trên dữ liệu BTC.
Chạy trực tiếp trong môi trường Python có đầy đủ GPU/PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Thêm root workspace vào sys.path để import BackEnd package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor
from BackEnd.app.shot_extractor.shot_extractor import ShotExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual Accuracy & Integration Test for Shot and Keyframe Extractor"
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default="L21_V001",
        help="ID video trong data/video/ (ví dụ: L21_V001)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["time", "hybrid_clip", "hybrid_clip_strict"],
        default="time",
        help="Chiến lược trích xuất keyframe: 'time' (phân bổ theo thời gian) hoặc 'hybrid_clip'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_result/manual_outputs/shot_keyframe",
        help="Thư mục xuất báo cáo và ảnh keyframe trích xuất",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất ranh giới shot của TransNetV2 (mặc định: 0.5)",
    )
    parser.add_argument(
        "--min-shot-duration-ms",
        type=int,
        default=500,
        help="Thời lượng shot tối thiểu (ms) (mặc định: 500)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_id = args.video_id
    output_dir = Path(args.output_dir) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" KIỂM THỬ THỦ CÔNG SHOT & KEYFRAME EXTRACTOR")
    print(f" - Video ID: {video_id}")
    print(f" - Keyframe Strategy: {args.strategy}")
    print(f" - Output Directory: {output_dir.resolve()}")
    print("=" * 80)

    # 1. Khởi tạo & Chạy ShotExtractor
    print("\n[1/2] Đang trích xuất ranh giới shot bằng TransNetV2...")
    try:
        shot_extractor = ShotExtractor(
            threshold=args.threshold,
            min_shot_duration_ms=args.min_shot_duration_ms,
        )
        shots = shot_extractor.extract(video_id)
        print(f" => Trích xuất thành công {len(shots)} shots cho video '{video_id}'.")
    except Exception as e:
        print(f" [LỖI] Trích xuất shot thất bại: {e}")
        sys.exit(1)

    # 2. Khởi tạo & Chạy KeyframeExtractor
    print(f"\n[2/2] Đang trích xuất keyframe bổ sung (Strategy: {args.strategy})...")
    try:
        keyframe_extractor = KeyframeExtractor(
            keyframe_dir=output_dir,
            strategy=args.strategy,
        )
        keyframes = keyframe_extractor.extract_for_video(video_id, shots)
        print(f" => Trích xuất thành công {len(keyframes)} keyframe bổ sung.")
    except Exception as e:
        print(f" [LỖI] Trích xuất keyframe thất bại: {e}")
        sys.exit(1)

    # 3. Hiển thị bảng tổng hợp kết quả
    print("\n" + "=" * 80)
    print(f" BẢNG TỔNG HỢP SHOTS & KEYFRAMES ({video_id})")
    print("=" * 80)
    print(f"{'Shot ID':<16} | {'Khoảng thời gian (ms)':<22} | {'Frame Range':<15} | {'Số Keyframe'}")
    print("-" * 80)

    shot_kf_map: dict[str, list[dict]] = {s.shot_id: [] for s in shots}
    for kf in keyframes:
        if kf.shot_id in shot_kf_map:
            shot_kf_map[kf.shot_id].append({
                "frame_id": kf.frame_id,
                "frame_idx": kf.frame_idx,
                "timestamp_ms": kf.timestamp_ms,
                "frame_path": str(kf.frame_path),
            })

    for shot in shots[:15]:  # Hiển thị tối đa 15 shot đầu
        kf_count = len(shot_kf_map.get(shot.shot_id, []))
        time_range = f"{shot.start_ms}ms -> {shot.end_ms}ms"
        frame_range = f"[{shot.start_frame_idx}, {shot.end_frame_idx}]"
        print(f"{shot.shot_id:<16} | {time_range:<22} | {frame_range:<15} | {kf_count}")

    if len(shots) > 15:
        print(f"... và {len(shots) - 15} shots khác.")

    # 4. Xuất file kết quả JSON
    report_data = {
        "video_id": video_id,
        "strategy": args.strategy,
        "total_shots": len(shots),
        "total_keyframes_extracted": len(keyframes),
        "shots": [
            {
                "shot_id": s.shot_id,
                "shot_index": s.shot_index,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "start_frame_idx": s.start_frame_idx,
                "end_frame_idx": s.end_frame_idx,
                "extracted_keyframes": shot_kf_map.get(s.shot_id, []),
            }
            for s in shots
        ],
    }

    json_report_path = output_dir / "shot_keyframe_summary.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f" [HOÀN TẤT] File báo cáo JSON: {json_report_path}")
    print(f" Các file ảnh keyframe được lưu tại: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
