\# 🚗 Vietnam License Plate Recognition (LPR)



Hệ thống nhận diện và đọc biển số xe Việt Nam theo thời gian thực, được thiết kế theo kiến trúc 2 giai đoạn (Detection + OCR) và tối ưu hóa để chạy mượt trên môi trường \*\*CPU-only (Edge / Low-resource devices)\*\* sử dụng \*\*YOLO11\*\*, \*\*RapidOCR (ONNX Runtime)\*\* và \*\*Streamlit\*\*.



!\[Demo](assets/demo.png)



\## ✨ Tính năng chính

\- \*\*Phát hiện biển số độ chính xác cao:\*\* Fine-tune mô hình `YOLO11n` trên tập dữ liệu biển số xe Việt Nam (`mAP@0.5 ≈ 0.99`, `mAP@0.5:0.95 ≈ 0.76`).

\- \*\*Đọc \& chuẩn hóa biển số Việt Nam:\*\* Tích hợp \*\*RapidOCR\*\* hỗ trợ cả biển số dài (1 dòng) và biển số vuông (2 dòng), kết hợp \*\*Regex post-processing\*\* tự động sửa các lỗi nhận diện ký tự phổ biến (`O/0`, `I/1`, `B/8`, `S/5`...) và chuẩn hóa định dạng (VD: `30K-555.55`, `29B1-123.45`).

\- \*\*Tối ưu cho máy không có GPU:\*\* Chạy trực tiếp trên CPU với tùy chọn chuyển đổi giữa các định dạng \*\*PyTorch (`.pt`)\*\*, \*\*ONNX (`.onnx`)\*\* và \*\*OpenVINO\*\*.

\- \*\*Giao diện Web trực quan:\*\* Xây dựng bằng \*\*Streamlit\*\*, hỗ trợ tải lên ảnh và video (tích hợp frame skipping để giảm tải CPU), hiển thị ảnh biển số đã cắt (crop) và xuất kết quả ra file \*\*CSV\*\*.



\## 🏗️ Kiến trúc hệ thống



```text

Input (Image / Video)

&#x20;  │

&#x20;  ▼

\[Stage 1] YOLO11n Detector (PyTorch / ONNX / OpenVINO)

&#x20;  │  └─ Phát hiện tọa độ Bounding Box biển số

&#x20;  ▼

Crop \& Resize (Pre-processing)

&#x20;  │

&#x20;  ▼

\[Stage 2] RapidOCR (ONNX Runtime)

&#x20;  │  └─ Sắp xếp tọa độ ký tự (hỗ trợ biển 1 dòng \& 2 dòng)

&#x20;  ▼

VN Plate Regex Formatter (Post-processing)

&#x20;  │  └─ Sửa lỗi ký tự \& chuẩn hóa định dạng biển số VN

&#x20;  ▼

Streamlit Web UI \& CSV Export

```



\## 📊 Kết quả huấn luyện \& Benchmark



\### 1. Quá trình huấn luyện (`YOLO11n`)

Mô hình được huấn luyện trên Google Colab (T4 GPU) với `imgsz=640`, `batch=16` và tự động dừng sớm (early stopping) khi loss hội tụ ổn định:



!\[Training Results](assets/results.png)



\### 2. Benchmark tốc độ suy luận trên CPU

Thực nghiệm đo trên máy tính cấu hình \*\*CPU Intel (không có GPU rời), 8GB RAM\*\*, chạy trung bình 30 lần (`warmup=5`, kích thước đầu vào `640x640`):



| Định dạng Model | Avg Latency (ms) | Tốc độ (FPS) | Ghi chú |

| :--- | :---: | :---: | :--- |

| \*\*PyTorch (`.pt`)\*\* | \*\*93.56 ms\*\* | \*\*10.69 FPS\*\* | \*\*Nhanh nhất trong thực nghiệm\*\* |

| \*\*ONNX (`.onnx`)\*\* | \*\*99.79 ms\*\* | \*\*10.02 FPS\*\* | Nhẹ, dễ deploy đa nền tảng không cần PyTorch |

| OpenVINO | 371.05 ms | 2.70 FPS | Chậm hơn trên cấu hình RAM/CPU giới hạn |



> \*\*Nhận xét:\*\* Trên cấu hình CPU và RAM hạn chế, \*\*PyTorch\*\* và \*\*ONNX Runtime\*\* cho hiệu năng ổn định (\~10 FPS), đủ để chạy mượt cho ứng dụng nhận diện ảnh và video kết hợp frame skipping.



\## 🚀 Hướng dẫn cài đặt \& Chạy ứng dụng



\### 1. Clone repository

```bash

git clone \[https://github.com/](https://github.com/)<username>/vietnam-license-plate-recognition.git

cd vietnam-license-plate-recognition

```



\### 2. Tạo môi trường ảo và cài đặt thư viện

```bash

python -m venv .venv



\# Windows (PowerShell):

.\\.venv\\Scripts\\Activate.ps1



\# Linux / macOS:

\# source .venv/bin/activate



pip install -r requirements.txt

```



\### 3. Chạy giao diện Web (Streamlit)

```bash

streamlit run app.py

```

Truy cập địa chỉ `http://localhost:8501` trên trình duyệt để sử dụng.



\### 4. Chạy đo hiệu năng CPU (Benchmark)

```bash

python benchmark.py

```



\## 📁 Cấu trúc thư mục



```text

vietnam-license-plate-recognition/

├── assets/                  # Ảnh demo giao diện và biểu đồ kết quả huấn luyện

│   ├── demo.png

│   └── results.png

├── notebooks/

│   └── train\_yolo.ipynb     # Notebook huấn luyện YOLO11n và export model trên Google Colab

├── src/

│   ├── detector.py          # Class phát hiện biển số bằng YOLO11

│   ├── ocr\_reader.py        # Class tiền xử lý ảnh, đọc ký tự RapidOCR \& chuẩn hóa Regex

│   └── pipeline.py          # Kết nối toàn bộ luồng Detection + OCR

├── weights/                 # Trọng số model sau khi huấn luyện (.pt, .onnx, openvino)

├── app.py                   # Giao diện web Streamlit (hỗ trợ Ảnh \& Video)

├── benchmark.py             # Script đo độ trễ (ms) và tốc độ (FPS) trên CPU

├── requirements.txt         # Danh sách thư viện cần thiết

└── README.md                # Tài liệu hướng dẫn dự án

```



\## 🛠️ Công nghệ sử dụng

\- \*\*Object Detection:\*\* Ultralytics YOLO11

\- \*\*OCR Engine:\*\* RapidOCR (ONNX Runtime)

\- \*\*Image Processing:\*\* OpenCV, NumPy

\- \*\*Web Interface:\*\* Streamlit, Pandas

\- \*\*Deployment / Export:\*\* PyTorch, ONNX, OpenVINO

