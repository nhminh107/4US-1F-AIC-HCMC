"""Script chạy thử nghiệm thực tế (Demo Script) cho Module 2.4 - Keyframe Extractor.

Chạy trích xuất keyframe bổ sung trực tiếp trên 1 video thật của BTC trong thư mục data/.
In báo cáo phân rã thời gian chạy từng phase chi tiết để làm cơ sở làm việc với các bên.

Sử dụng:
    python scripts/run_extract_keyframe_demo.py --video_id L21_V001
    python scripts/run_extract_keyframe_demo.py --video_id L21_V001 --real_shot
    python scripts/run_extract_keyframe_demo.py --video_id L21_V001 --real_shot --strategy hybrid_clip
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.app.contracts.pipeline import ShotMetadata
from BackEnd.app.keyframe_extractor.config import HybridKeyframeConfig
from BackEnd.app.keyframe_extractor.keyframe_extractor import KeyframeExtractor
from BackEnd.app.keyframe_extractor import frame_decoder as frame_decoder_module
from BackEnd.app.shot_extractor.shot_extractor import DEFAULT_WEIGHTS_PATH, ShotExtractor


def run_demo(
    video_id: str = "L21_V001",
    use_real_shot_extractor: bool = False,
    *,
    strategy: str = "time",
    sample_fps: float = 1.0,
    max_candidate_frames: int = 64,
    ffmpeg_batch_size: int = 100,
    max_shots: int | None = None,
    output_dir: Path | None = None,
    diagnostics: bool = False,
    shot_extractor: ShotExtractor | None = None,
    keyframe_extractor: KeyframeExtractor | None = None,
) -> tuple[ShotExtractor | None, KeyframeExtractor | None]:
    print("=" * 75)
    print(f"[BENCHMARK RUNNER] Starting extraction breakdown for video: {video_id}")
    print(f"[BENCHMARK RUNNER] Keyframe strategy: {strategy}")
    print(f"[BENCHMARK RUNNER] Hybrid config: sample_fps={sample_fps}, max_candidate_frames={max_candidate_frames}")
    print(f"[BENCHMARK RUNNER] FFmpeg export batch size: {ffmpeg_batch_size}")
    print("=" * 75)

    if diagnostics:
        _print_runtime_diagnostics()

    t_start = time.time()

    # 1. Đọc danh sách Keyframe Official thực tế do BTC cấp
    t0 = time.time()
    official_dir = PROJECT_ROOT / "data" / "Keyframes_L21" / "keyframes" / video_id
    official_frame_idxs: list[int] = []
    if official_dir.is_dir():
        for img_file in official_dir.glob("*.jpg"):
            try:
                idx = int(img_file.stem)
                official_frame_idxs.append(idx)
            except ValueError:
                pass
    t_official = time.time() - t0
    print(f"1. Official Keyframe Load: {len(official_frame_idxs)} images loaded ({t_official:.3f}s)")

    # 2. Lấy danh sách Shot (Chạy ShotExtractor thật hoặc dùng Shot mẫu)
    t0 = time.time()
    if use_real_shot_extractor:
        print("2. Running TransNetV2 ShotExtractor model (Module 2.3 - Lộc)...")
        if shot_extractor is None:
            weights_path = (
                DEFAULT_WEIGHTS_PATH
                if DEFAULT_WEIGHTS_PATH.is_file()
                else PROJECT_ROOT / "docs_rule_diagram" / "transnetv2-pytorch-weights.pth"
            )
            shot_extractor = ShotExtractor(weights_path=weights_path)
        shots = shot_extractor.extract(video_id)
        print(f"   -> Detected {len(shots)} real shots from TransNetV2.")
    else:
        print("2. Using sample shots for fast FFmpeg decoding verification...")
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
            ShotMetadata(
                shot_id=f"{video_id}_S003",
                video_id=video_id,
                shot_index=2,
                start_ms=20000,
                end_ms=40000,
                start_frame_idx=500,
                end_frame_idx=999,
            ),
        ]
        print(f"   -> Initialized {len(shots)} sample shots for testing.")
    shots = _limit_shots(shots, max_shots)
    if max_shots is not None:
        print(f"   -> Limited benchmark to {len(shots)} shot(s).")
    t_shot = time.time() - t0

    # 3. Chọn keyframe theo từng shot, sau đó export JPEG ở cấp toàn video
    t0 = time.time()
    out_keyframe_dir = output_dir or PROJECT_ROOT / "data" / "keyframes"
    video_output_dir = _prepare_output_dir(out_keyframe_dir, video_id)
    if keyframe_extractor is None:
        keyframe_extractor = KeyframeExtractor(
            keyframe_dir=out_keyframe_dir,
            strategy=strategy,
            hybrid_config=HybridKeyframeConfig(
                sample_fps=sample_fps,
                max_candidate_frames_per_shot=max_candidate_frames,
            ),
            max_frames_per_ffmpeg_batch=ffmpeg_batch_size,
        )

    print("3. Running KeyframeExtractor video-level FFmpeg export (Module 2.4 - Trường)...")
    print(f"   -> Output directory: {video_output_dir}")
    sparse_official = [idx for idx in official_frame_idxs if idx % 50 == 0]
    extracted_metadatas = keyframe_extractor.extract_for_video(
        video_id=video_id,
        shots=shots,
        existing_frame_idxs=sparse_official,
        progress_callback=_print_progress_event,
    )
    if diagnostics:
        selector = extractor.hybrid_selector
        selector_label = type(selector).__name__ if selector is not None else "not initialized"
        selector_id = id(selector) if selector is not None else "n/a"
        print(f"   -> Hybrid selector instance: {selector_label} ({selector_id})")
    t_keyframe = time.time() - t0

    t_total = time.time() - t_start

    # Tính phần trăm thời gian
    pct_official = (t_official / t_total) * 100 if t_total > 0 else 0
    pct_shot = (t_shot / t_total) * 100 if t_total > 0 else 0
    pct_keyframe = (t_keyframe / t_total) * 100 if t_total > 0 else 0

    # 4. In báo cáo phân rã thời gian chuyên nghiệp
    print("\n" + "=" * 75)
    print("📊 BÁO CÁO PHÂN RÃ THỜI GIAN THỰC THI (DETAILED BENCHMARK REPORT)")
    print("=" * 75)
    print(f"Target Video: {video_id}.mp4 | Total Shots: {len(shots)} | Keyframes Extracted: {len(extracted_metadatas)}")
    print("-" * 75)
    print(f"{'Giai đoạn (Pipeline Phase)':<42} | {'Thời gian (s)':<12} | {'Tỷ lệ (%)':<10}")
    print("-" * 75)
    print(f"{'1. Nạp Keyframe Official BTC':<42} | {t_official:>10.2f}s | {pct_official:>8.1f}%")
    
    import torch

    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    shot_label = f"2. Cut Shot TransNetV2 [{device_name}]"
    print(f"{shot_label:<42} | {t_shot:>10.2f}s | {pct_shot:>8.1f}%")
    
    keyframe_label = f"3. Keyframe Extraction FFmpeg [{strategy}]"
    print(f"{keyframe_label:<42} | {t_keyframe:>10.2f}s | {pct_keyframe:>8.1f}%")
    print("-" * 75)
    print(f"{'TỔNG THỜI GIAN TOÀN BỘ PIPELINE':<42} | {t_total:>10.2f}s | {100.0:>8.1f}%")
    print("=" * 75)

    if extracted_metadatas:
        print("\n📸 Mẫu 3 Keyframe bổ sung đầu tiên:")
        for meta in extracted_metadatas[:3]:
            print(
                f"   * ID: {meta.frame_id:<14} | Index: {meta.frame_idx:<5} | "
                f"Time: {meta.timestamp_ms:<6}ms | Resolution: {meta.width}x{meta.height} | Path: {meta.frame_path.name}"
            )

    print("\n✅ HOÀN THÀNH CHẠY BENCHMARK DỮ LIỆU THỰC TẾ 100%!")
    return shot_extractor, keyframe_extractor


def _prepare_output_dir(output_dir: Path, video_id: str) -> Path:
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = video_output_dir / ".write_test.tmp"
    try:
        probe_path.write_text("ok", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Output directory is not writable: {video_output_dir}") from error
    finally:
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            pass
    return video_output_dir


def _limit_shots(shots: list[ShotMetadata], max_shots: int | None) -> list[ShotMetadata]:
    if max_shots is None:
        return shots
    if max_shots <= 0:
        raise ValueError("max_shots must be positive.")
    return sorted(shots, key=lambda shot: shot.shot_index)[:max_shots]


def _print_progress_event(event: dict[str, object]) -> None:
    if event.get("phase") == "start":
        return
    if event.get("phase") == "export":
        print(
            "   -> "
            f"📦 Final FFmpeg export GPU | frames={event['frame_count']} | "
            f"chunks={event['chunk_count']} | batch_size={event['chunk_size']} | "
            f"export={float(event['export_s']):.2f}s",
            flush=True,
        )
        return
    hybrid = event.get("hybrid")
    if isinstance(hybrid, dict) and hybrid:
        print(
            "   -> "
            f"Shot {event['shot_number']:>3}/{event['total_shots']} [{event['shot_id']}] | "
            f"Candidates: {hybrid.get('candidate_count', 0):>2} | "
            f"Selected: {event['selected_count']:>2} | "
            f"Thời gian: {float(event['elapsed_s']):>5.2f}s "
            f"(RAM Decode: {float(hybrid.get('decode_s', 0.0)):.2f}s | "
            f"CLIP AI: {float(hybrid.get('clip_s', 0.0)):.2f}s | "
            f"Cluster: {float(hybrid.get('cluster_s', 0.0)):.2f}s | "
            f"Dedup: {float(hybrid.get('redundancy_s', 0.0)):.2f}s)",
            flush=True,
        )
        return
    print(
        "   -> "
        f"Shot {event['shot_number']:>3}/{event['total_shots']} [{event['shot_id']}] | "
        f"Selected: {event['selected_count']:>2} | Thời gian: {float(event['elapsed_s']):.2f}s",
        flush=True,
    )


def _print_runtime_diagnostics() -> None:
    source = inspect.getsource(frame_decoder_module.extract_and_save_frames)
    chunked_source = inspect.getsource(frame_decoder_module.extract_and_save_frames_chunked)
    has_mkdir_guard = "mkdir(parents=True, exist_ok=True)" in source
    has_chunked_export = "extract_and_save_frames(" in chunked_source
    print("[DIAGNOSTICS]")
    print(f"   frame_decoder module: {frame_decoder_module.__file__}")
    print(f"   extract_and_save_frames mkdir guard: {has_mkdir_guard}")
    print(f"   extract_and_save_frames_chunked available: {has_chunked_export}")
    print(f"   python executable: {sys.executable}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo trích xuất keyframe thực tế trên 1 video BTC")
    parser.add_argument(
        "--video_id",
        type=str,
        nargs="+",
        default=["L21_V001"],
        help="Danh sách mã video_id cần trích xuất (ví dụ: --video_id L21_V001 L21_V002)",
    )
    parser.add_argument(
        "--real_shot", action="store_true", help="Chạy TransNetV2 thật để cắt shot (tốn CPU/GPU)"
    )
    parser.add_argument(
        "--strategy",
        choices=("time", "hybrid_clip", "hybrid_clip_strict"),
        default="time",
        help="Chiến lược chọn keyframe bổ sung.",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Số candidate frames mỗi giây cho hybrid_clip.",
    )
    parser.add_argument(
        "--max_candidate_frames",
        type=int,
        default=64,
        help="Giới hạn số candidate frames mỗi shot cho hybrid_clip.",
    )
    parser.add_argument(
        "--ffmpeg_batch_size",
        type=int,
        default=100,
        help="Số frame tối đa mỗi batch FFmpeg khi export JPEG toàn video.",
    )
    parser.add_argument(
        "--max_shots",
        type=int,
        default=None,
        help="Giới hạn số shot đầu tiên để benchmark nhanh.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "keyframes",
        help="Thư mục gốc để ghi keyframes extracted.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="In thông tin module/runtime để kiểm tra import path.",
    )

    args = parser.parse_args()
    shared_shot_extractor = None
    shared_keyframe_extractor = None

    for vid in args.video_id:
        shared_shot_extractor, shared_keyframe_extractor = run_demo(
            video_id=vid,
            use_real_shot_extractor=args.real_shot,
            strategy=args.strategy,
            sample_fps=args.sample_fps,
            max_candidate_frames=args.max_candidate_frames,
            ffmpeg_batch_size=args.ffmpeg_batch_size,
            max_shots=args.max_shots,
            output_dir=args.output_dir,
            diagnostics=args.diagnostics,
            shot_extractor=shared_shot_extractor,
            keyframe_extractor=shared_keyframe_extractor,
        )
