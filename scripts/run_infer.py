import argparse
import sys
from pathlib import Path
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.detector import PPEDetector

def main():
    parser = argparse.ArgumentParser(description="Run PPE Detection on Image/Video/Webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt or .onnx model")
    parser.add_argument("--source", type=str, default="0", help="Path to image, video, or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display visual results in OpenCV window")
    parser.add_argument("--save", type=str, default="", help="Path to save output result")
    args = parser.parse_args()

    detector = PPEDetector(model_path=args.model, conf_threshold=args.conf)

    source = args.source
    is_webcam = source.isdigit()

    if is_webcam or Path(source).suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
        cap = cv2.VideoCapture(int(source) if is_webcam else source)
        if not cap.isOpened():
            print(f"[!] Error: Could not open video source {source}")
            return

        print("[*] Processing video feed. Press 'q' to quit.")
        writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            annotated = detector.draw_annotations(frame, detections)

            violations = [d for d in detections if d["is_violation"]]
            cv2.putText(
                annotated,
                f"Violations: {len(violations)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255) if violations else (0, 255, 0),
                2
            )

            if args.save and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 25, (w, h))

            if writer:
                writer.write(annotated)

            if args.show:
                cv2.imshow("PPE Mining Compliance Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    else:
        # Single image
        img = cv2.imread(source)
        if img is None:
            print(f"[!] Error: Unable to read image {source}")
            return

        detections = detector.detect(img)
        annotated = detector.draw_annotations(img, detections)
        violations = [d for d in detections if d["is_violation"]]
        print(f"[+] Total detected: {len(detections)}, Violations: {len(violations)}")

        if args.save:
            cv2.imwrite(args.save, annotated)
            print(f"[+] Saved output to {args.save}")

        if args.show:
            cv2.imshow("PPE Detection Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
