"""
Hau xu ly cho Recognition (muc 5, 7 module_ocr.md):
- Sap xep cac vung chu trong 1 anh theo thu tu doc tu nhien (tren->duoi, trai->phai)
  dua tren bbox, truoc khi gop thanh mang `texts` cuoi cung.
- Chuan hoa chuoi text doc duoc (xoa khoang trang thua) va loai trung lap.
"""
import re
from typing import Dict, List

_WHITESPACE_RE = re.compile(r"\s+")

# 2 vung co y_min lech nhau trong khoang nay (px, sau khi da resize ve max_side=960)
# duoc coi la cung 1 "hang" chu, de tranh sap sai thu tu do chenh lech nho giua cac
# bbox tren cung 1 dong (vd dau/khong dau).
_LINE_Y_TOLERANCE = 15


def normalize_text(text: str) -> str:
    """Xoa khoang trang thua o dau/cuoi va gom nhieu khoang trang lien tiep thanh 1."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def sort_regions_reading_order(regions: List[Dict]) -> List[Dict]:
    """
    Sap xep danh sach region (moi region co "bbox": [x_min, y_min, x_max, y_max])
    theo thu tu doc tu nhien: gom theo hang dua tren y_min (cho phep sai lech nho
    trong _LINE_Y_TOLERANCE), trong cung hang sap theo x_min tang dan.
    """

    def sort_key(region: Dict):
        x_min, y_min = region["bbox"][0], region["bbox"][1]
        return (round(y_min / _LINE_Y_TOLERANCE), x_min)

    return sorted(regions, key=sort_key)


def merge_texts(texts: List[str]) -> List[str]:
    """
    Loc bo chuoi rong sau chuan hoa va loai trung lap (giu nguyen thu tu xuat hien
    dau tien) truoc khi dua vao `texts` cuoi cung cua 1 keyframe.
    """
    merged: List[str] = []
    seen = set()
    for text in texts:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged
