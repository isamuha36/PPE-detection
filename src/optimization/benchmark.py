import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2

def benchmark_model(
    model_path: str,
    img_size: int = 640,
    warmup_runs: int = 20,
    test_runs: int = 100,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Benchmarks model inference latency, FPS, and stability.
    Supports PyTorch (.pt) and ONNX (.onnx).
    """
    from ultralytics import YOLO
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[*] Loading model: {model_path} on {device}")
    model = YOLO(str(model_path))

    # Dummy test image
    dummy_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    print(f"[*] Warming up for {warmup_runs} iterations...")
    for _ in range(warmup_runs):
        _ = model.predict(dummy_input, device=device, verbose=False)

    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()

    latencies = []
    print(f"[*] Running {test_runs} benchmark iterations...")
    for _ in range(test_runs):
        start = time.perf_counter()
        _ = model.predict(dummy_input, device=device, verbose=False)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    metrics = {
        "model": model_path.name,
        "device": device,
        "input_resolution": f"{img_size}x{img_size}",
        "iterations": test_runs,
        "mean_latency_ms": round(mean_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "fps": round(fps, 1),
    }

    if torch.cuda.is_available() and "cuda" in device:
        metrics["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
        metrics["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)

    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Benchmark for TRPL Engineering Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt or .onnx model")
    parser.add_argument("--img-size", type=int, default=640, help="Input image dimension")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    args = parser.parse_args()

    results = benchmark_model(args.model, img_size=args.img_size, test_runs=args.runs, device=args.device)
    print("\n" + "=" * 50)
    print("BENCHMARK REPORT (TRPL AI & DATA SCIENCE STANDARDS)")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k:25}: {v}")
    print("=" * 50)
