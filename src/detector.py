import cv2
import numpy as np
from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self, model_path: str = "weights/best_openvino_model", conf_threshold: float = 0.4):
        """
        Khởi tạo model YOLO. Mặc định dùng OpenVINO để tối ưu tốc độ trên CPU Intel.
        """
        self.model = YOLO(model_path, task="detect")
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray):
        """
        Phát hiện biển số trong ảnh BGR.
        Trả về list các dict: {'bbox': [x1, y1, x2, y2], 'conf': float, 'crop': np.ndarray}
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        h, w = image.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # Giới hạn tọa độ trong khung ảnh để tránh lỗi khi cắt
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "crop": crop
                })

        return detections