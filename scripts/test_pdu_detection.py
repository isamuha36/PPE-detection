import cv2
from pathlib import Path
from src.inference.detector import PPEDetector

def run_test():
    img_path = Path("data/samples/pdu_frame_60.jpg")
    if not img_path.exists():
        print("Sample frame not found")
        return

    detector = PPEDetector("models/trained/best.pt", conf_threshold=0.25)
    image = cv2.imread(str(img_path))
    detections = detector.detect(image)

    print(f"Total detections on PDU frame: {len(detections)}")
    for d in detections:
        print(f"  Class: {d['class_name']:12} Conf: {d['confidence']:.2f} Bbox: {[round(v, 1) for v in d['bbox']]}")

    annotated = detector.draw_annotations(image, detections)
    out_path = Path("data/samples/pdu_frame_60_detected.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"[+] Saved annotated frame to {out_path}")

if __name__ == "__main__":
    run_test()
