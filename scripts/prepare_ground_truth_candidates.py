import argparse
import json
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detector import PPEDetector

CLASS_NAMES = [
    "person",
    "cap_on",
    "cap_off",
    "mask_on",
    "mask_off",
    "jacket_on",
    "jacket_off",
    "gloves_on",
    "gloves_off"
]

def prepare_ground_truth(
    video_dir: str = "data/raw/pdu_videos",
    output_dir: str = "data/pdu_ground_truth",
    target_count: int = 50,
    imgsz: int = 1280,
    conf_threshold: float = 0.25
):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visualized"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Save classes.txt for annotation tools
    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as cf:
        for c in CLASS_NAMES:
            cf.write(f"{c}\n")

    videos = sorted(list(video_dir.glob("*.mp4")))
    if not videos:
        print(f"[!] No videos found in {video_dir}")
        return

    frames_per_video = target_count // len(videos)
    print(f"[*] Preparing {target_count} Ground Truth candidate frames ({frames_per_video} per video)...")

    detector = PPEDetector("models/trained/best.pt", conf_threshold=conf_threshold)

    saved_total = 0

    for v_idx, video_path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Spread samples evenly across the video
        start_frame = int(total_frames * 0.02)
        end_frame = int(total_frames * 0.90)
        usable_range = end_frame - start_frame
        step_size = max(1, usable_range // frames_per_video)

        print(f"\n[{v_idx}/{len(videos)}] Processing {video_path.name}: {total_frames} frames ({total_frames / (fps * 60):.1f} min)")
        print(f"  Target: {frames_per_video} frames spaced by ~{step_size / (fps * 60):.1f} min ({step_size} frames)")

        frame_idx = 0
        v_saved = 0

        while frame_idx < end_frame and v_saved < frames_per_video:
            if frame_idx >= start_frame and (frame_idx - start_frame) % step_size == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                time_min = (frame_idx / fps) / 60.0
                base_name = f"pdu_gt_v{v_idx}_f{frame_idx:07d}_min{int(time_min):03d}"

                # Run inference with high-res 1280
                results = detector.model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=conf_threshold,
                    verbose=False
                )[0]

                boxes_yolo = []
                frame_detections = []

                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = results.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    xywhn = box.xywhn[0].tolist()
                    xyxy = box.xyxy[0].tolist()

                    boxes_yolo.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                    frame_detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": xyxy,
                        "is_violation": cls_name.endswith("_off") or cls_name.startswith("no_")
                    })

                # Save clean image
                img_path = images_dir / f"{base_name}.jpg"
                cv2.imwrite(str(img_path), frame)

                # Save draft label
                lbl_path = labels_dir / f"{base_name}.txt"
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    lf.writelines(boxes_yolo)

                # Save visualized preview
                vis_img = detector.draw_annotations(frame.copy(), frame_detections)
                vis_path = vis_dir / f"{base_name}_annotated.jpg"
                cv2.imwrite(str(vis_path), vis_img)

                v_saved += 1
                saved_total += 1
                print(f"  [+] Saved {v_saved}/{frames_per_video} ({base_name}.jpg, {len(boxes_yolo)} draft boxes) at minute {time_min:.1f}")

            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

        cap.release()

    print(f"\n============================================================")
    print(f"   Candidate Dataset Prepared Successfully!")
    print(f"============================================================")
    print(f"Total Frames: {saved_total}")
    print(f"Images: {images_dir}")
    print(f"Draft Labels: {labels_dir}")
    print(f"Visualized Previews: {vis_dir}")
    print(f"Classes Definition: {classes_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 50 diverse Ground Truth candidate frames from PDU videos")
    parser.add_argument("--count", type=int, default=50, help="Total candidate frames (default: 50)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image resolution for draft inference (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    prepare_ground_truth(target_count=args.count, imgsz=args.imgsz, conf_threshold=args.conf)
