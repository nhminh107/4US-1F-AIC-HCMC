"""
Text Recognition cho Module OCR (muc 3, 5, 7 module_ocr.md).

Input: ket qua Detection cua 1 chunk (list tra ve boi run_detection_on_chunk(), giu
nguyen trong RAM - KHONG doc/ghi lai tu file), moi phan tu la 1 keyframe kem
detected_regions, moi region da co san anh crop dang numpy array ("cropped_image").

Xu ly theo dung mo hinh muc 5: gop TOAN BO crop cua ca chunk vao 1 danh sach, dedup
bang Perceptual Hashing (dedup.py) truoc, roi Batch Recognition - goi VietOCR theo
tung batch (predict_batch) thay vi goi predict() le tung crop. Sau cung, sap xep lai
text theo bbox (postprocess.py) truoc khi gop thanh `texts` cho tung anh.
"""
from typing import Dict, List, Optional

from BackEnd.CONFIG import OCR_RECOGNITION_BATCH_SIZE as BATCH_SIZE
from BackEnd.app.ocr.recognition.dedup import DedupCache, is_duplicate_hash
from BackEnd.app.ocr.recognition.postprocess import merge_texts, sort_regions_reading_order
from BackEnd.app.ocr.recognition.recognizer import TextRecognizer

def run_recognition_on_chunk(
    detection_results: List[Dict],
    recognizer: TextRecognizer,
    dedup_cache: DedupCache,
    batch_size: int = BATCH_SIZE,
    stats: Optional[Dict] = None,
) -> List[Dict]:
    # Gop toan bo region (kem anh crop) cua ca chunk vao 1 danh sach phang, giu lai
    # item_index de map nguoc ket qua ve dung anh sau khi Recognition xong.
    flat_regions: List[Dict] = []
    for item_index, item in enumerate(detection_results):
        for region in item["detected_regions"]:
            flat_regions.append({"item_index": item_index, "region": region})

    region_texts: List[Optional[str]] = [None] * len(flat_regions)

    # Buoc 1 - Dedup bang Perceptual Hashing: voi tung region, tra cache dung chung
    # (tu cac chunk truoc) truoc; neu chua co, gom nhom voi cac region KHAC trong
    # cung chunk nay co hash gan trung (pending_reps) - vi cache chi duoc cap nhat
    # SAU khi Batch Recognition chay xong, neu khong gom nhom truoc thi 2 vung giong
    # het nhau nhung cung lan dau gap trong 1 chunk se bi Recognition 2 lan rieng biet.
    pending_reps: List[Dict] = []  # {"hash": int, "image": ndarray, "flat_indices": [int, ...]}
    for i, entry in enumerate(flat_regions):
        cropped = entry["region"]["cropped_image"]
        if cropped is None or cropped.size == 0:
            region_texts[i] = ""
            continue

        region_hash = dedup_cache.compute_hash(cropped)
        cached_text = dedup_cache.find(region_hash)
        if cached_text is not None:
            region_texts[i] = cached_text
            continue

        matched_rep = next(
            (rep for rep in pending_reps if is_duplicate_hash(rep["hash"], region_hash)),
            None,
        )
        if matched_rep is not None:
            matched_rep["flat_indices"].append(i)
        else:
            pending_reps.append({"hash": region_hash, "image": cropped, "flat_indices": [i]})

    # Buoc 2 - Batch Recognition: goi VietOCR theo tung batch_size, 1 lan goi cho MOI
    # nhom vung khac nhau (representative) thay vi goi predict() le tung crop.
    for start in range(0, len(pending_reps), batch_size):
        batch_reps = pending_reps[start : start + batch_size]
        batch_images = [rep["image"] for rep in batch_reps]

        batch_results = recognizer.recognize_batch(batch_images)
        for rep, (text, _confidence) in zip(batch_reps, batch_results):
            dedup_cache.add(rep["hash"], text)
            for flat_i in rep["flat_indices"]:
                region_texts[flat_i] = text

    # Buoc 3 - Hau xu ly: gom text da doc duoc ve dung tung anh, sap xep theo bbox
    # (thu tu doc tu nhien) roi chuan hoa/loai trung truoc khi tra ve `texts`.
    regions_per_item: List[List[Dict]] = [[] for _ in detection_results]
    for i, entry in enumerate(flat_regions):
        regions_per_item[entry["item_index"]].append(
            {**entry["region"], "text": region_texts[i] or ""}
        )

    results: List[Dict] = []
    for item, regions in zip(detection_results, regions_per_item):
        ordered_regions = sort_regions_reading_order(regions)
        texts = merge_texts([region["text"] for region in ordered_regions])
        results.append({"img_path": item["img_path"], "texts": texts})

    if stats is not None:
        recognized = len(pending_reps)
        stats["total_regions"] = stats.get("total_regions", 0) + len(flat_regions)
        stats["recognized"] = stats.get("recognized", 0) + recognized
        stats["reused_from_cache"] = stats.get("reused_from_cache", 0) + (
            len(flat_regions) - recognized
        )

    return results


if __name__ == "__main__":
    from BackEnd.app.ocr.detection.run_detection import run_detection

    _recognizer = TextRecognizer()
    _cache = DedupCache()
    print(run_recognition_on_chunk(run_detection(), _recognizer, _cache))
