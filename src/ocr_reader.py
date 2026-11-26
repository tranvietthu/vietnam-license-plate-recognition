import re
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class LicensePlateOCR:
    def __init__(self):
        self.ocr = RapidOCR()

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Phóng to ảnh nếu quá nhỏ và tăng độ tương phản nhẹ."""
        h, w = crop.shape[:2]
        if h < 64:
            scale = 64 / h
            crop = cv2.resize(crop, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)
        return crop

    def clean_and_format_vn_plate(self, raw_text: str) -> str:
        """Chuẩn hóa ký tự và sửa lỗi OCR phổ biến cho biển số Việt Nam."""
        # Chỉ giữ lại chữ cái in hoa và chữ số
        text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if len(text) < 7 or len(text) > 9:
            return text

        # Bảng chuyển đổi lỗi OCR hay nhầm
        char_to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
        digit_to_char = {"0": "D", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

        chars = list(text)

        # 2 ký tự đầu (mã tỉnh) luôn là số
        for i in [0, 1]:
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

        # Ký tự thứ 3 (series) luôn là chữ cái
        if chars[2] in digit_to_char:
            chars[2] = digit_to_char[chars[2]]

        # 4 hoặc 5 ký tự cuối luôn là số
        for i in range(4, len(chars)):
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]

        cleaned = "".join(chars)

        # Định dạng đẹp (VD: 29A-123.45 hoặc 29A1-123.45)
        if len(cleaned) == 8:  # Biển ô tô 5 số: 30F12345 -> 30F-123.45
            return f"{cleaned[:3]}-{cleaned[3:6]}.{cleaned[6:]}"
        elif len(cleaned) == 9:  # Biển xe máy 5 số: 29B112345 -> 29B1-123.45
            return f"{cleaned[:4]}-{cleaned[4:7]}.{cleaned[7:]}"
        elif len(cleaned) == 7:  # Biển 4 số: 29A1234 -> 29A-1234
            return f"{cleaned[:3]}-{cleaned[3:]}"

        return cleaned

    def read(self, crop: np.ndarray):
        """
        Đọc biển số từ ảnh crop.
        Trả về tuple: (text_da_chuan_hoa, confidence_trung_binh)
        """
        processed = self.preprocess(crop)
        result, _ = self.ocr(processed)

        if not result:
            return "", 0.0

        # Sắp xếp các cụm text theo thứ tự từ trên xuống dưới (hỗ trợ biển 2 dòng), sau đó trái sang phải
        # Mỗi phần tử trong result có dạng: [box, text, score]
        result.sort(key=lambda item: (item[0][0][1] // 20, item[0][0][0]))

        raw_text = "".join([item[1] for item in result])
        avg_conf = float(np.mean([float(item[2]) for item in result]))

        formatted_text = self.clean_and_format_vn_plate(raw_text)
        return formatted_text, avg_conf