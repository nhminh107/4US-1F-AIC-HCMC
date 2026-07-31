"""
Text Recognition wrapper quanh VietOCR (checkpoint vgg_transformer).
Nhan vao 1 anh da cat san (vung chu, numpy array BGR tu OpenCV) va doc noi dung.

Ly do dung VietOCR thay vi phan recognition co san cua PaddleOCR: VietOCR la model
chuyen biet tieng Viet, doc chinh xac dau cau hon (xem muc 2 module_ocr.md).
"""
from typing import List, Tuple

import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


class TextRecognizer:
    def __init__(self, model_name: str = "vgg_transformer", device: str = "cpu"):
        # Load cau hinh mac dinh cua VietOCR ung voi checkpoint da chon.
        config = Cfg.load_config_from_name(model_name)
        # Chay tren CPU theo mac dinh (doi sang "cuda:0" neu may co GPU).
        config["device"] = device
        # Khoi tao predictor tu cau hinh tren, tu dong tai weight pretrained ve.
        self._predictor = Predictor(config)

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Doc noi dung chu trong 1 anh da cat san.
        Tra ve (text, confidence). Neu anh rong (crop bi loi) tra ve ("", 0.0).
        """
        # Anh crop co the rong neu bbox bi cat het sau khi gioi han trong kich thuoc anh goc.
        if image is None or image.size == 0:
            return "", 0.0

        # VietOCR nhan PIL Image he mau RGB, trong khi anh dau vao la numpy BGR (OpenCV).
        pil_image = Image.fromarray(image[:, :, ::-1])
        # return_prob=True de lay kem do tin cay cua ket qua doc duoc.
        text, prob = self._predictor.predict(pil_image, return_prob=True)
        return text, float(prob)

    def recognize_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Doc noi dung chu cho ca 1 danh sach anh trong 1 lan goi VietOCR
        (predict_batch), thay vi goi predict() le tung anh (muc 5 module_ocr.md).

        Anh rong (bbox bi cat het) duoc loai ra truoc khi goi predict_batch (VietOCR
        khong xu ly duoc anh rong), ket qua tuong ung duoc dien lai ("", 0.0) theo
        dung vi tri ban dau trong danh sach tra ve.
        """
        results: List[Tuple[str, float]] = [("", 0.0)] * len(images)

        valid_indices = [i for i, image in enumerate(images) if image is not None and image.size > 0]
        if not valid_indices:
            return results

        pil_images = [Image.fromarray(images[i][:, :, ::-1]) for i in valid_indices]
        texts, probs = self._predictor.predict_batch(pil_images, return_prob=True)

        for index, text, prob in zip(valid_indices, texts, probs):
            results[index] = (text, float(prob))

        return results
