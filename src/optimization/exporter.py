from pathlib import Path
from typing import Optional
from ultralytics import YOLO

def export_model(
    model_path: str,
    format_type: str = "onnx",
    img_size: int = 640,
    half: bool = True,
    device: str = "0"
) -> Path:
    """
    Exports a trained PyTorch model (.pt) to ONNX or TensorRT (.engine) for accelerated edge inference.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model {model_path} does not exist")

    print(f"[*] Loading model {model_path} for export to {format_type.upper()}...")
    model = YOLO(str(model_file))

    exported_path = model.export(
        format=format_type,
        imgsz=img_size,
        half=half,
        device=device,
        simplify=True if format_type == "onnx" else False,
    )
    print(f"[+] Successfully exported model to: {exported_path}")
    return Path(exported_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export YOLO model for edge acceleration")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"], help="Export format")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 half precision")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    export_model(args.model, format_type=args.format, img_size=args.img_size, half=args.half)
