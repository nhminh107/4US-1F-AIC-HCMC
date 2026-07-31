"""
Script chay thu Text Detection tren 1 tap con cua data/keyframes/ that, de kiem tra
nhanh pipeline Detection chay dung va so luong vung chu detect duoc (bbox/det_confidence)
co hop ly khong, ma khong can chay Recognition/toan bo pipeline.

Anh crop khong ghi ra file (muc 3 module_ocr.md), chi giu trong RAM - script nay chi
in thong ke (so anh, so vung chu detect duoc theo tung anh).

Cach chay:
    python -m BackEnd.app.ocr.detection.test_detection
    python -m BackEnd.app.ocr.detection.test_detection --video K01_V001
    python -m BackEnd.app.ocr.detection.test_detection --video K01_V001 --shot-min 0 --shot-max 21
    python -m BackEnd.app.ocr.detection.test_detection --limit 20
"""
import argparse
import re
import time
from typing import List, Optional

from BackEnd.app.ocr.detection.detector import TextDetector
from BackEnd.app.ocr.detection.run_detection import KEYFRAMES_DIR, run_detection_on_chunk, scan_keyframes

_SHOT_NUM_RE = re.compile(r"S(\d+)")


def _shot_number(img_path: str) -> Optional[int]:
    # "K01_V001/S00001/F000025.jpg" -> 1
    shot_part = img_path.split("/")[1]
    match = _SHOT_NUM_RE.fullmatch(shot_part)
    return int(match.group(1)) if match else None


def filter_keyframes(
    img_paths: List[str],
    video: Optional[str] = None,
    shot_min: Optional[int] = None,
    shot_max: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[str]:
    filtered = img_paths
    if video:
        filtered = [p for p in filtered if p.startswith(f"{video}/")]
    if shot_min is not None or shot_max is not None:
        kept = []
        for img_path in filtered:
            shot_num = _shot_number(img_path)
            if shot_num is None:
                continue
            if shot_min is not None and shot_num < shot_min:
                continue
            if shot_max is not None and shot_num > shot_max:
                continue
            kept.append(img_path)
        filtered = kept
    if limit:
        filtered = filtered[:limit]
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Test Detection tren 1 tap con cua data/keyframes/")
    parser.add_argument("--video", type=str, default=None, help="Chi chay tren 1 video, vd K01_V001")
    parser.add_argument("--shot-min", type=int, default=None, help="Chi lay tu shot so (vd 0)")
    parser.add_argument("--shot-max", type=int, default=None, help="Chi lay den shot so (vd 21)")
    parser.add_argument("--limit", type=int, default=None, help="Gioi han so frame de test nhanh")
    args = parser.parse_args()

    img_paths = filter_keyframes(
        scan_keyframes(KEYFRAMES_DIR),
        video=args.video,
        shot_min=args.shot_min,
        shot_max=args.shot_max,
        limit=args.limit,
    )
    if not img_paths:
        print("Khong tim thay anh nao trong data/keyframes/ khop dieu kien.")
        return

    print(f"[test_detection] Chay Detection tren {len(img_paths)} frame tu data/keyframes/ ...")
    start = time.time()

    detector = TextDetector()
    results = run_detection_on_chunk(img_paths, detector, KEYFRAMES_DIR)

    total_regions = sum(len(item["detected_regions"]) for item in results)
    elapsed = time.time() - start

    print(f"[test_detection] Hoan tat sau {elapsed:.1f}s: {len(results)} anh, {total_regions} vung chu.")
    for item in results[:20]:
        print(f"  {item['img_path']}: {len(item['detected_regions'])} vung")


if __name__ == "__main__":
    main()
