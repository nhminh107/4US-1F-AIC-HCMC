"""
Text Detection cho Module OCR (muc 1, 3, 5 module_ocr.md).

Dau vao: quet truc tiep thu muc `data/keyframes/` (khong phu thuoc metadata cua
Module Embedding/Database - metadata do chi sinh ra SAU khi embedding chay xong,
con OCR phai chay duoc doc lap tren toan bo keyframe tu dau).

Detection chay theo tung Chunk keyframe (mac dinh CHUNK_SIZE=100, xem run_ocr_pipeline.py):
voi moi anh trong chunk, tien xu ly (resize/tang contrast) roi chay Detection lan luot,
loc theo det_confidence >= 0.6-0.7, cat cac vung chu giu dang numpy array TRONG RAM
(khong ghi file tam - xem muc 3) de truyen thang cho Recognition.
"""
import os
from typing import Dict, List, Optional

from BackEnd.CONFIG import (
    KEYFRAME_OUTPUT_DIR as KEYFRAMES_DIR,
    OCR_LEGACY_DETECTION_CONFIDENCE_THRESHOLD as DET_CONFIDENCE_THRESHOLD,
)
from BackEnd.app.ocr.detection.crop_utils import crop_region
from BackEnd.app.ocr.detection.detector import TextDetector
from BackEnd.app.ocr.detection.preprocess import preprocess

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def scan_keyframes(keyframes_dir: str = KEYFRAMES_DIR) -> List[str]:
    """
    Quet toan bo `data/keyframes/` (muc 1: "dau vao la toan bo anh nam trong thu
    muc data/keyframes/"), tra ve danh sach img_path tuong doi (vd
    "K01_V001/S00001/F000025.jpg"), da sap xep theo thu tu ten.
    """
    img_paths = []
    for root, _dirs, files in os.walk(keyframes_dir):
        for fname in files:
            if not fname.lower().endswith(_IMAGE_EXTENSIONS):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), keyframes_dir)
            img_paths.append(rel_path.replace(os.sep, "/"))
    img_paths.sort()
    return img_paths


def run_detection_on_chunk(
    img_paths: List[str],
    detector: TextDetector,
    keyframes_dir: str = KEYFRAMES_DIR,
    det_confidence_threshold: float = DET_CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    """
    Chay Detection lan luot cho tung anh trong 1 chunk (muc 5). Voi moi anh, tra ve
    danh sach cac vung chu da qua loc nguong, moi vung kem san anh crop dang numpy
    array ("cropped_image") de Recognition dung truc tiep, khong doc lai tu dia.

    Anh khong co vung chu nao dat nguong -> detected_regions rong [] (muc 3).
    """
    results: List[Dict] = []

    for img_path in img_paths:
        image_path = os.path.join(keyframes_dir, img_path)
        image = preprocess(image_path)

        raw_regions = detector.detect(image)

        detected_regions: List[Dict] = []
        region_idx = 0
        for region in raw_regions:
            # Bo qua vung co do tin cay thap hon nguong.
            if region["det_confidence"] < det_confidence_threshold:
                continue
            region_idx += 1
            region_id = f"r{region_idx:03d}"

            # Cat anh theo bbox, giu lai dang numpy array trong RAM (khong ghi file).
            cropped = crop_region(image, region["bbox"])

            detected_regions.append(
                {
                    "region_id": region_id,
                    "bbox": region["bbox"],
                    "det_confidence": region["det_confidence"],
                    "cropped_image": cropped,
                }
            )

        results.append({"img_path": img_path, "detected_regions": detected_regions})

    return results


def run_detection(
    img_paths: Optional[List[str]] = None,
    keyframes_dir: str = KEYFRAMES_DIR,
) -> List[Dict]:
    """
    Chay Detection khong chia chunk, tren toan bo (hoac 1 danh sach chi dinh) anh -
    tien cho viec goi le/test thu, khong dung cho pipeline chinh (xem
    run_ocr_pipeline.py, noi Detection duoc goi theo tung chunk).
    """
    detector = TextDetector()
    img_paths = img_paths if img_paths is not None else scan_keyframes(keyframes_dir)
    results = run_detection_on_chunk(img_paths, detector, keyframes_dir)

    total_regions = sum(len(item["detected_regions"]) for item in results)
    print(f"[run_detection] total={len(results)} regions={total_regions}")
    return results


if __name__ == "__main__":
    run_detection()
