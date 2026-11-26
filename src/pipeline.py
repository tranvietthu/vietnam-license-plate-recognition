import cv2
import numpy as np
from src.detector import LicensePlateDetector
from src.ocr_reader import LicensePlateOCR


class LPRPipeline:
    def __init__(self, model_path: str = "weights/best_openvino_model", conf_threshold: float = 0.4):
        self.detector = LicensePlateDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.ocr = LicensePlateOCR()

    def process_image(self, image: np.ndarray):
        """
        Xử lý 1 ảnh BGR: phát hiện biển số, đọc ký tự và vẽ kết quả.
        Trả về: (annotated_image, results_list)
        """
        annotated = image.copy()
        detections = self.detector.detect(image)
        results = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            plate_text, ocr_conf = self.ocr.read(det["crop"])

            results.append({
                "bbox": det["bbox"],
                "det_conf": round(det["conf"], 3),
                "text": plate_text if plate_text else "UNKNOWN",
                "ocr_conf": round(ocr_conf, 3),
                "crop": det["crop"]
            })

            # Vẽ khung và nhãn lên ảnh
            label = plate_text if plate_text else f"Plate {det['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

        return annotated, results


if __name__ == "__main__":
    
    test_img_path = "assets/img1.png"
    img = cv2.imread(test_img_path)

    if img is None:
        print(f"Không tìm thấy ảnh tại {test_img_path}. Hãy copy 1 ảnh xe vào thư mục assets/ để test.")
    else:
        pipeline = LPRPipeline(conf_threshold=0.4)
        out_img, res = pipeline.process_image(img)
        for r in res:
            print(f"Biển số: {r['text']} | Det conf: {r['det_conf']} | OCR conf: {r['ocr_conf']}")
        cv2.imwrite("assets/test_output.jpg", out_img)
        print("Đã lưu ảnh kết quả tại assets/test_output.jpg")