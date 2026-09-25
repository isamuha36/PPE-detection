import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Mining PPE Compliance Monitoring API & Dashboard",
    description="Real-time Personal Protective Equipment (APD) Detection Service for Mining Sites (TRPL SV UGM x PT Parama Data Unit)",
    version="1.0.0"
)

# Setup templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global detector state
current_detector = None
current_model_name = "best.onnx"
active_violations_count = 0

def get_detector(model_choice: Optional[str] = None):
    global current_detector, current_model_name
    target_name = model_choice or current_model_name

    if current_detector is None or (model_choice and model_choice != current_model_name):
        candidates = [
            Path(f"models/exported/{target_name}"),
            Path(f"models/trained/{target_name}"),
            Path("models/exported/best.onnx"),
            Path("models/trained/best.pt"),
            Path("models/pretrained/yolov8n.pt")
        ]
        chosen = None
        for p in candidates:
            if p.exists():
                chosen = p
                break

        if not chosen:
            raise HTTPException(
                status_code=500,
                detail=f"Model {target_name} not found. Ensure models/exported/best.onnx or models/trained/best.pt exist."
            )

        from src.inference.detector import PPEDetector
        current_detector = PPEDetector(model_path=chosen)
        current_model_name = chosen.name
        print(f"[*] Initialized detector with {chosen}")

    return current_detector

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Serves the interactive real-time monitoring web dashboard.
    """
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "PPE Detection API",
        "active_model": current_model_name
    }

@app.get("/api/v1/stats")
def get_stats():
    return {
        "active_model": current_model_name,
        "active_violations": active_violations_count
    }

@app.post("/api/v1/model/switch")
def switch_model(model_name: str = Query("best.onnx", choices=["best.onnx", "best.pt"])):
    det = get_detector(model_name)
    return {"message": "Model switched successfully", "active_model": current_model_name}

import random
from collections import defaultdict

# Pre-indexed test samples by scenario
test_samples_cache = defaultdict(list)

def get_test_samples():
    global test_samples_cache
    if not test_samples_cache:
        test_img_dir = Path("data/processed/test/images")
        test_lbl_dir = Path("data/processed/test/labels")
        if test_img_dir.exists() and test_lbl_dir.exists():
            for lf in test_lbl_dir.glob("*.txt"):
                try:
                    with open(lf, "r", encoding="utf-8") as f:
                        lbl_classes = set()
                        for line in f:
                            p = line.strip().split()
                            if p:
                                lbl_classes.add(int(p[0]))
                    img_match = list(test_img_dir.glob(f"{lf.stem}.*"))
                    if img_match:
                        img_path = img_match[0]
                        test_samples_cache["all"].append(img_path)
                        if 2 in lbl_classes:  # cap_off (helmet violation)
                            test_samples_cache["cap_off"].append(img_path)
                        if 8 in lbl_classes:  # gloves_off (glove violation)
                            test_samples_cache["gloves_off"].append(img_path)
                        if (1 in lbl_classes or 5 in lbl_classes) and 2 not in lbl_classes and 8 not in lbl_classes:
                            test_samples_cache["compliant"].append(img_path)
                except Exception:
                    continue
    return test_samples_cache

@app.get("/api/v1/sample/worker")
def get_sample_worker(category: str = Query("random", choices=["random", "all", "compliant", "cap_off", "gloves_off"])):
    """
    Returns a random test image based on the selected scenario category.
    """
    cache = get_test_samples()
    key = "all" if category == "random" else category
    candidates = cache.get(key, []) or cache.get("all", [])

    if not candidates:
        fallback = Path("data/samples/sample_worker_1.jpg")
        if fallback.exists():
            with open(fallback, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg", headers={"X-Filename": fallback.name, "X-Category": "sample"})
        raise HTTPException(status_code=404, detail="No test samples found")

    selected_img = random.choice(candidates)
    with open(selected_img, "rb") as f:
        return Response(
            content=f.read(),
            media_type="image/jpeg",
            headers={
                "X-Filename": selected_img.name,
                "X-Category": category,
                "Access-Control-Expose-Headers": "X-Filename, X-Category"
            }
        )

@app.post("/api/v1/detect")
async def detect_ppe(
    file: UploadFile = File(...),
    conf: float = Query(0.45, ge=0.1, le=1.0)
):
    """
    Detects PPE items from an uploaded image file, measures actual inference latency, and returns structured data.
    """
    global active_violations_count
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    det_engine = get_detector()
    det_engine.conf_threshold = conf

    # Measure exact inference duration
    t_start = time.perf_counter()
    detections = det_engine.detect(image)
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)
    fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0

    violations = [d for d in detections if d.get("is_violation", False)]
    active_violations_count = len(violations)

    return JSONResponse({
        "filename": file.filename,
        "active_model": current_model_name,
        "inference_time_ms": latency_ms,
        "fps": fps,
        "total_detections": len(detections),
        "total_violations": len(violations),
        "compliant": len(violations) == 0,
        "detections": detections
    })

@app.post("/api/v1/detect/visualize")
async def detect_and_visualize(
    file: UploadFile = File(...),
    conf: float = Query(0.45, ge=0.1, le=1.0)
):
    """
    Detects PPE items and returns the annotated image directly (JPEG) with latency headers.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    det_engine = get_detector()
    det_engine.conf_threshold = conf

    t_start = time.perf_counter()
    detections = det_engine.detect(image)
    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)
    fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0

    annotated = det_engine.draw_annotations(image, detections)

    # Overlay latency watermark on annotated preview
    watermark = f"{current_model_name} | {latency_ms}ms ({fps} FPS)"
    cv2.putText(annotated, watermark, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    _, encoded_img = cv2.imencode(".jpg", annotated)
    return Response(
        content=encoded_img.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Inference-Time-Ms": str(latency_ms),
            "X-FPS": str(fps),
            "X-Active-Model": current_model_name,
            "Access-Control-Expose-Headers": "X-Inference-Time-Ms, X-FPS, X-Active-Model"
        }
    )

def generate_video_stream():
    """
    Generator function for MJPEG video streaming with real-time PPE detection.
    Falls back gracefully to simulated video if webcam is unavailable.
    """
    global active_violations_count
    cap = cv2.VideoCapture(0)
    is_live = cap.isOpened()

    sample_img = None
    if not is_live:
        sample_path = Path("data/samples/sample_worker_1.jpg")
        if sample_path.exists():
            sample_img = cv2.imread(str(sample_path))

    det_engine = get_detector()

    try:
        while True:
            if is_live:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if sample_img is not None:
                    frame = sample_img.copy()
                    cv2.putText(
                        frame,
                        "SIMULATED MINING CCTV (CAM-01)",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                    time.sleep(0.04)  # ~25 FPS
                else:
                    break

            detections = det_engine.detect(frame)
            annotated = det_engine.draw_annotations(frame, detections)
            violations = [d for d in detections if d.get("is_violation", False)]
            active_violations_count = len(violations)

            # Draw status banner
            status_text = f"VIOLATIONS: {len(violations)}" if violations else "STATUS: COMPLIANT"
            status_color = (0, 0, 255) if violations else (0, 255, 0)
            cv2.putText(
                annotated,
                status_text,
                (15, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if is_live:
            cap.release()

@app.get("/api/v1/stream/webcam")
def stream_webcam():
    """
    Live video MJPEG feed with real-time detection overlay.
    """
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=True)
