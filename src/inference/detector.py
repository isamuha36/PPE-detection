from pathlib import Path
from typing import List, Dict, Any, Union
import cv2
import numpy as np

class PPEDetector:
    """
    Unified PPE Detector supporting both PyTorch (.pt) and exported ONNX (.onnx) formats.
    """
    def __init__(self, model_path: Union[str, Path], conf_threshold: float = 0.45, iou_threshold: float = 0.45):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Lazy import ultralytics
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single BGR image.
        Returns a list of detected bounding boxes and metadata.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]

                # Violation flag based on class name
                is_violation = cls_name.endswith("_off") or cls_name.startswith("no_")

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": xyxy,
                    "is_violation": is_violation
                })

        return detections

    def draw_annotations(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto the image.
        Green for compliance (_on), Red for violation (_off).
        """
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]
            is_violation = det["is_violation"]

            # Red for violations, Green for compliance, Blue for person
            if cls_name == "person":
                color = (255, 165, 0)
            elif is_violation:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated
