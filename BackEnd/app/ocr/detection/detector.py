"""
Text Detection wrapper quanh PaddleOCR (module TextDetection - PP-OCRv4_mobile_det).
Chi khoanh vung (bounding box) noi co chu, khong doc noi dung.

Luu y: dung module `TextDetection` rieng (khong phai class `PaddleOCR` full pipeline)
vi tu paddleocr 3.x, class `PaddleOCR` luon chay ca detection + recognition, khong con
tham so `det=True, rec=False` nhu ban 2.x. `TextDetection` la module detection-only
tuong ung cho kien truc 2 buoc (Detection - Nguoi 1 / Recognition - Nguoi 2) cua nhom.

`engine_config={"enable_new_ir": False}` de tranh loi noi bo cua Paddle PIR executor
tren CPU/Windows voi model PP-OCRv4 hien tai (NotImplementedError khi chay engine mac dinh).
"""
from typing import Dict, List

import numpy as np
from paddleocr import TextDetection


class TextDetector:
    def __init__(self, model_name: str = "PP-OCRv4_mobile_det"):
        # Khoi tao model TextDetection cua PaddleOCR voi ten model duoc chon.
        # engine_config chi dinh chay engine "paddle" (static) va tat new_ir, de
        # tranh loi NotImplementedError tren Windows/CPU.
        self._det = TextDetection(
            model_name=model_name,
            engine_config={"run_mode": "paddle", "enable_new_ir": False},
        )

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Chay text detection tren mot anh (numpy array, BGR).

        Tra ve danh sach dict: {"bbox": [x_min, y_min, x_max, y_max], "det_confidence": float}
        """
        # Danh sach ket qua se tra ve, moi phan tu la 1 vung phat hien duoc.
        regions: List[Dict] = []
        # Chay model tren anh dau vao, lap qua tung ket qua tra ve (thuong chi co 1).
        for result in self._det.predict(image):
            # Lay danh sach polygon (toa do 4 diem goc cua tung vung chu) va diem tin cay tuong ung.
            polys = result.get("dt_polys", [])
            scores = result.get("dt_scores", [])
            # Duyet qua tung cap (polygon, score).
            for poly, score in zip(polys, scores):
                # Lay ra toan bo toa do x va toa do y cua 4 diem trong polygon.
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                # Tinh bounding box thang truc bang cach lay gia tri x, y nho nhat/lon nhat.
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                # Them ket qua (bbox + do tin cay) vao danh sach tra ve.
                regions.append({"bbox": bbox, "det_confidence": float(score)})

        return regions
