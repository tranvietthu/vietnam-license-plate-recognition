import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from src.pipeline import LPRPipeline

st.set_page_config(page_title="Vietnam LPR - YOLO11 & OpenVINO", layout="wide")

st.title(" Vietnam License Plate Recognition (LPR)")
st.markdown("Hệ thống nhận diện biển số xe Việt Nam tối ưu chạy trên **CPU** với **YOLO11 (OpenVINO)** & **RapidOCR**.")

# Sidebar cấu hình
st.sidebar.header(" Cấu hình")
model_option = st.sidebar.selectbox(
    "Chọn định dạng Model",
    ("weights/best_openvino_model", "weights/best.onnx", "weights/best.pt"),
    index=0
)
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.1, 0.9, 0.4, 0.05)


@st.cache_resource
def load_pipeline(model_path, conf):
    return LPRPipeline(model_path=model_path, conf_threshold=conf)


pipeline = load_pipeline(model_option, conf_threshold)
pipeline.detector.conf_threshold = conf_threshold

tab1, tab2 = st.tabs([" Nhận diện Ảnh", " Nhận diện Video"])

# --- TAB 1: XỬ LÝ ẢNH ---
with tab1:
    uploaded_img = st.file_uploader("Tải lên ảnh xe (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        start_time = time.time()
        annotated_img, results = pipeline.process_image(image)
        elapsed_ms = (time.time() - start_time) * 1000

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption=f"Kết quả ({elapsed_ms:.1f} ms)", use_container_width=True)

        with col2:
            st.subheader(" Kết quả chi tiết")
            if results:
                table_data = []
                for idx, r in enumerate(results, 1):
                    st.image(cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB), caption=f"Biển #{idx}: {r['text']}", width=200)
                    table_data.append({
                        "STT": idx,
                        "Biển số": r["text"],
                        "Det Conf": r["det_conf"],
                        "OCR Conf": r["ocr_conf"]
                    })
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    " Tải kết quả (CSV)",
                    df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="lpr_image_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Không phát hiện được biển số nào. Hãy thử hạ ngưỡng Confidence ở thanh bên trái.")

# --- TAB 2: XỬ LÝ VIDEO ---
with tab2:
    uploaded_vid = st.file_uploader("Tải lên video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Bỏ qua frame (tăng để chạy nhanh hơn trên CPU)", 1, 10, 3)

    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())

        if st.button(" Bắt đầu nhận diện Video"):
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            detected_plates = {}
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                annotated_frame, results = pipeline.process_image(frame)
                for r in results:
                    text = r["text"]
                    if text != "UNKNOWN" and len(text) >= 7:
                        # Lưu lại lần đọc có độ tin cậy OCR cao nhất
                        if text not in detected_plates or r["ocr_conf"] > detected_plates[text]["OCR Conf"]:
                            detected_plates[text] = {
                                "Biển số": text,
                                "Det Conf": r["det_conf"],
                                "OCR Conf": r["ocr_conf"],
                                "Frame": frame_idx
                            }

                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap.release()
            st.success("Hoàn thành xử lý video!")

            if detected_plates:
                df_vid = pd.DataFrame(list(detected_plates.values()))
                st.dataframe(df_vid, use_container_width=True, hide_index=True)
                st.download_button(
                    "Tải danh sách biển số (CSV)",
                    df_vid.to_csv(index=False).encode("utf-8-sig"),
                    file_name="lpr_video_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Không đọc được biển số hợp lệ nào trong video.")