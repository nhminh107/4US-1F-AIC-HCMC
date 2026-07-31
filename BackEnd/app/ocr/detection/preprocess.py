"""
OpenCV preprocessing utilities cho buoc Text Detection: resize, tang contrast
(muc 5 module_ocr.md).

Luu y: text tu logo/watermark KHONG bi loai bo - nhom chap nhan giu lai cac text nay
trong ket qua OCR, nen khong ap dung buoc che vung co dinh.
"""
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    # Doc anh tu duong dan bang OpenCV.
    image = cv2.imread(image_path)
    # Neu doc that bai (file khong ton tai/hong), cv2 tra ve None -> bao loi ro rang.
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def resize_image(image: np.ndarray, max_side: int = 960) -> np.ndarray:
    # Lay chieu cao, chieu rong cua anh.
    h, w = image.shape[:2]
    # Tinh ty le thu nho de canh dai nhat bang max_side.
    scale = max_side / max(h, w)
    # Neu anh da nho hon hoac bang max_side thi khong lam gi, giu nguyen anh.
    if scale >= 1.0:
        return image
    # Nguoc lai, thu nho anh theo ty le da tinh (giu nguyen ty le khung hinh).
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    # Chuyen anh tu he mau BGR sang he mau LAB.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Tach rieng 3 kenh: L (do sang), a va b (mau sac).
    l_channel, a_channel, b_channel = cv2.split(lab)
    # Tao bo loc CLAHE (can bang histogram thich ung, gioi han do tuong phan).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Ap dung CLAHE len rieng kenh do sang L de tang tuong phan.
    l_channel = clahe.apply(l_channel)
    # Ghep lai 3 kenh (L da tang tuong phan + a, b giu nguyen).
    lab = cv2.merge((l_channel, a_channel, b_channel))
    # Chuyen nguoc ve he mau BGR de tra ve.
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(image_path: str) -> np.ndarray:
    # Doc anh tu file.
    image = load_image(image_path)
    # Thu nho anh neu qua lon (canh dai nhat 960px).
    image = resize_image(image)
    # Tang tuong phan cho anh.
    image = enhance_contrast(image)
    return image
