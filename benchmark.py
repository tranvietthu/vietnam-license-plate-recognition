import time
import cv2
import numpy as np
from ultralytics import YOLO

MODELS = {
    "PyTorch (.pt)": "weights/best.pt",
    "ONNX (.onnx)": "weights/best.onnx",
    "OpenVINO": "weights/best_openvino_model",
}

def run_benchmark(image_path: str = None, runs: int = 30, warmup: int = 5):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = None


    if img is None:
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print(f"=== CPU Inference Benchmark ({runs} runs, warmup={warmup}) ===\n")
    print(f"{'Format':<18} | {'Avg Latency (ms)':<18} | {'FPS':<10}")
    print("-" * 52)

    for name, path in MODELS.items():
        model = YOLO(path, task="detect")

        # Warmup để ổn định CPU cache
        for _ in range(warmup):
            model.predict(img, verbose=False)

        start = time.perf_counter()
        for _ in range(runs):
            model.predict(img, verbose=False)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / runs) * 1000
        fps = runs / elapsed
        print(f"{name:<18} | {avg_ms:<18.2f} | {fps:<10.2f}")

if __name__ == "__main__":
    run_benchmark(image_path="assets/img1.png")