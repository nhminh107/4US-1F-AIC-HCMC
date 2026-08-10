"""
Dedup vung crop bang Perceptual Hashing (muc 5 module_ocr.md).

Vi nhom chap nhan giu lai thong tin watermark/logo dai (khong crop bo), cac vung nay
thuong lap lai giong het nhau qua nhieu keyframe. DedupCache tinh pHash cho tung vung
crop va giu 1 cache (hash -> text) dung chung xuyen suot ca lan chay (khong chi trong
1 chunk), cho phep tai su dung lai ket qua Recognition cu thay vi goi lai VietOCR.
"""
from typing import List, Optional

import cv2
import numpy as np

from BackEnd.CONFIG import (
    OCR_DEDUP_HAMMING_DISTANCE_THRESHOLD as HAMMING_DISTANCE_THRESHOLD,
)


def perceptual_hash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Tinh average-hash (aHash) cho 1 anh: thu nho ve grayscale hash_size x hash_size,
    so tung diem anh voi gia tri trung binh -> chuoi bit -> so nguyen.
    Anh giong nhau (hoac gan giong) se cho ra hash giong nhau/gan nhau.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    avg = float(resized.mean())

    h = 0
    for value in resized.flatten():
        h = (h << 1) | int(value > avg)
    return h


def hamming_distance(hash_a: int, hash_b: int) -> int:
    # So bit khac nhau giua 2 so nguyen (XOR roi dem so bit 1).
    return bin(hash_a ^ hash_b).count("1")


def is_duplicate_hash(hash_a: int, hash_b: int, threshold: int = HAMMING_DISTANCE_THRESHOLD) -> bool:
    return hamming_distance(hash_a, hash_b) <= threshold


class DedupCache:
    """
    Cache (hash, text) cac vung crop da tung duoc Recognition doc qua. Duyet tuyen
    tinh de so sanh Hamming distance khi tra cuu - chap nhan duoc vi so logo/watermark
    khac nhau thuc te la it, du so keyframe/crop co the rat lon.
    """

    def __init__(self, threshold: int = HAMMING_DISTANCE_THRESHOLD):
        self._threshold = threshold
        self._entries: List[dict] = []

    def compute_hash(self, image: np.ndarray) -> int:
        return perceptual_hash(image)

    def find(self, region_hash: int) -> Optional[str]:
        """Tra ve text da cache neu co vung nao gan trung (Hamming distance), None neu chua gap."""
        for entry in self._entries:
            if is_duplicate_hash(entry["hash"], region_hash, self._threshold):
                return entry["text"]
        return None

    def add(self, region_hash: int, text: str) -> None:
        self._entries.append({"hash": region_hash, "text": text})

    def __len__(self) -> int:
        return len(self._entries)
