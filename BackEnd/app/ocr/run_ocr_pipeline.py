"""
Script tong cua Module OCR (muc 3, 5, 6, 7 module_ocr.md).

Detection va Recognition chay noi tiep nhau trong cung 1 tien trinh Python (muc 3):
anh crop giu dang numpy array trong RAM, khong ghi file tam. De giai quyet nghen co
chai giua 2 buoc (muc 5), toan bo keyframe trong data/keyframes/ duoc chia thanh
cac Chunk (CHUNK_SIZE keyframe/dot); voi moi chunk, Detection chay lan luot tren ca
chunk, roi toan bo crop thu duoc duoc dedup bang Perceptual Hashing va dua qua
Batch Recognition (batch_size) thay vi goi model le tung crop. Khong dung
multiprocessing de tranh khac biet hanh vi giua Windows/Linux khi cac may trong
nhom chay khac he dieu hanh.

Ket qua cua tung chunk duoc ghi (cong don) vao ocr_final_results.json ngay sau khi
xu ly xong chunk do, truoc khi chuyen sang chunk tiep theo - neu tien trinh bi dung
giua chung, ket qua cac chunk da xong truoc do khong bi mat.
"""
import json
import os
from typing import Dict, List, Optional

from BackEnd.app.ocr.detection.detector import TextDetector
from BackEnd.app.ocr.detection.run_detection import KEYFRAMES_DIR, run_detection_on_chunk, scan_keyframes
from BackEnd.app.ocr.recognition.dedup import DedupCache
from BackEnd.app.ocr.recognition.recognizer import TextRecognizer
from BackEnd.app.ocr.recognition.run_recognition import BATCH_SIZE, run_recognition_on_chunk

# So keyframe xu ly Detection truoc khi gop crop dua qua Recognition (muc 5) - du nho
# de khong giu qua nhieu crop trong RAM cung luc, du lon de Batch Recognition phat huy
# hieu qua. Nen test lai tren may se dung luc thi de chon cau hinh on dinh.
CHUNK_SIZE = 100

_OCR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(_OCR_DIR, "output", "ocr_final_results.json")


def _iter_chunks(items: List[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _write_json(results: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_pipeline(
    keyframes_dir: str = KEYFRAMES_DIR,
    output_path: str = OUTPUT_PATH,
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = BATCH_SIZE,
    img_paths: Optional[List[str]] = None,
) -> List[Dict]:
    # Dau vao: toan bo anh trong data/keyframes/ (hoac danh sach duoc chi dinh rieng,
    # tien cho viec test tren 1 tap con).
    img_paths = img_paths if img_paths is not None else scan_keyframes(keyframes_dir)

    # Khoi tao 2 model 1 lan duy nhat, dung lai cho toan bo cac chunk.
    detector = TextDetector()
    recognizer = TextRecognizer()
    # Cache pHash dung chung xuyen suot ca lan chay (khong reset theo chunk), vi
    # logo/watermark co the lap lai xuyen suot ca video, khong chi trong 1 chunk.
    dedup_cache = DedupCache()

    stats: Dict = {"total_regions": 0, "recognized": 0, "reused_from_cache": 0}
    all_results: List[Dict] = []

    chunks = list(_iter_chunks(img_paths, chunk_size))
    total_chunks = len(chunks)

    for chunk_idx, chunk in enumerate(chunks, start=1):
        detection_results = run_detection_on_chunk(chunk, detector, keyframes_dir)
        chunk_results = run_recognition_on_chunk(
            detection_results, recognizer, dedup_cache, batch_size=batch_size, stats=stats
        )
        all_results.extend(chunk_results)

        # Ghi cong don ra file JSON tong ngay sau khi xong 1 chunk (muc 5).
        _write_json(all_results, output_path)

        print(
            f"[run_pipeline] chunk {chunk_idx}/{total_chunks} ({len(chunk)} keyframe) "
            f"xong -> tong {len(all_results)} keyframe da xu ly"
        )

    print(
        f"[run_pipeline] Hoan tat: {len(all_results)} keyframe, "
        f"total_regions={stats['total_regions']} recognized={stats['recognized']} "
        f"reused_from_cache={stats['reused_from_cache']} -> {output_path}"
    )
    return all_results


if __name__ == "__main__":
    run_pipeline()
