import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from src.inference.detector import PPEDetector

class PDUVideoExtractor:
    """
    Extracts frames from PDU industrial CCTV videos and performs automated
    pseudo-labeling using a pre-trained PPE detection model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/trained/best.pt",
        output_dir: Union[str, Path] = "data/pdu_curated",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.vis_dir = self.output_dir / "visualized"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PPEDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        self.conf_threshold = conf_threshold

    def process_video(
        self,
        video_path: Union[str, Path],
        interval_seconds: float = 10.0,
        max_frames: Optional[int] = None,
        save_visualizations: bool = True,
        filter_empty: bool = True,
        video_prefix: Optional[str] = None
    ) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """
        Processes a single video sequentially, samples keyframes,
        generates YOLO label annotations, and saves images/visualizations.

        Args:
            video_path: Path to the .mp4 video file.
            interval_seconds: Time interval between extracted frames in seconds.
            max_frames: Maximum number of frames to extract (useful for testing/batching).
            save_visualizations: Whether to save bounding-box annotated frames for human review.
            filter_empty: If True, only frames with at least one detection are saved.
            video_prefix: Optional prefix for image filenames (defaults to video stem).

        Returns:
            Dictionary with extraction statistics.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(round(fps * interval_seconds)))
        stem = video_prefix or video_path.stem

        print(f"[*] Processing {video_path.name}: {total_video_frames} frames @ {fps:.1f} FPS")
        print(f"[*] Sampling 1 frame every {interval_seconds}s (step={frame_step} frames)")

        stats = {
            "video_name": video_path.name,
            "total_video_frames": total_video_frames,
            "sampled_frames": 0,
            "saved_frames": 0,
            "empty_frames_skipped": 0,
            "class_distribution": {},
            "elapsed_seconds": 0.0
        }

        t_start = time.time()
        frame_idx = 0
        extracted_count = 0

        while True:
            if max_frames and extracted_count >= max_frames:
                print(f"[!] Reached max_frames limit: {max_frames}")
                break

            if frame_idx % frame_step == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                stats["sampled_frames"] += 1
                extracted_count += 1
                time_sec = frame_idx / fps

                # Run inference via detector's YOLO model
                results = self.detector.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.detector.iou_threshold,
                    verbose=False
                )

                # Parse detections
                boxes_yolo = []
                frame_detections = []
                r = results[0]
                h_img, w_img = frame.shape[:2]

                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names.get(cls_id, str(cls_id))
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

                    stats["class_distribution"][cls_name] = stats["class_distribution"].get(cls_name, 0) + 1

                # Filter empty frames
                if filter_empty and len(boxes_yolo) == 0:
                    stats["empty_frames_skipped"] += 1
                else:
                    base_name = f"{stem}_f{frame_idx:07d}_t{int(time_sec):05d}"
                    img_file = self.images_dir / f"{base_name}.jpg"
                    lbl_file = self.labels_dir / f"{base_name}.txt"

                    # Save image and YOLO annotation
                    cv2.imwrite(str(img_file), frame)
                    with open(lbl_file, "w", encoding="utf-8") as lf:
                        lf.writelines(boxes_yolo)

                    # Save annotated visualization for human-in-the-loop review
                    if save_visualizations and len(frame_detections) > 0:
                        vis_img = self.detector.draw_annotations(frame.copy(), frame_detections)
                        vis_file = self.vis_dir / f"{base_name}_annotated.jpg"
                        cv2.imwrite(str(vis_file), vis_img)

                    stats["saved_frames"] += 1

                if extracted_count % 25 == 0:
                    print(f"  -> Extracted {extracted_count} samples ({frame_idx}/{total_video_frames} frames, saved {stats['saved_frames']})")

            else:
                # Fast forward to next keyframe without decoding raster
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

        cap.release()
        stats["elapsed_seconds"] = round(time.time() - t_start, 2)
        print(f"[+] Finished {video_path.name}: {stats['saved_frames']} frames saved in {stats['elapsed_seconds']}s")
        return stats
