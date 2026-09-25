import shutil
from pathlib import Path
from ultralytics import YOLO

def train_ppe_model(
    data_yaml: str = "configs/data.yaml",
    model_type: str = "yolo11s.pt",
    epochs: int = 40,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project_name: str = "models/trained",
    run_name: str = "yolo_ppe_enhanced"
):
    """
    Trains an enhanced YOLO model for mining PPE compliance detection.
    Applies spatial attention backbone (YOLO11s) and augmentations tailored for small object APD detection.
    """
    print(f"[*] Initializing model training: {model_type} on device {device}")
    model = YOLO(model_type)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_name,
        name=run_name,
        save=True,
        save_period=5,
        plots=True,
        workers=4,
        cache=True,        # Cache dataset in RAM to eliminate disk I/O bottleneck
        patience=15,       # Early stopping patience
        cls=1.2,           # Increased classification loss weight to distinguish violation vs compliance
        cos_lr=True,       # Cosine learning rate scheduling
        mosaic=1.0,        # Multi-image context mosaic
        mixup=0.12,        # Image blending for robustness
        close_mosaic=10,   # Disable mosaic for last 10 epochs for fine localization
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,      # Slight rotation augmentation
        fliplr=0.5,        # Horizontal flip
        scale=0.5,         # Multi-scale augmentation
    )

    best_checkpoint = Path(model.trainer.best)
    print(f"[+] Training completed. Best checkpoint saved at {best_checkpoint}")

    # Stage best model to standard paths
    if best_checkpoint.exists():
        trained_dir = Path("models/trained")
        exported_dir = Path("models/exported")
        trained_dir.mkdir(parents=True, exist_ok=True)
        exported_dir.mkdir(parents=True, exist_ok=True)

        target_pt = trained_dir / "best.pt"
        shutil.copy2(best_checkpoint, target_pt)
        print(f"[+] Staged best PyTorch model to: {target_pt}")

        # Export to ONNX for edge deployment
        print("[*] Exporting best model to ONNX...")
        best_model = YOLO(str(target_pt))
        exported_onnx = best_model.export(
            format="onnx",
            imgsz=img_size,
            simplify=True
        )
        if exported_onnx and Path(exported_onnx).exists():
            target_onnx = trained_dir / "best.onnx"
            exported_onnx_path = exported_dir / "best.onnx"
            if Path(exported_onnx).resolve() != target_onnx.resolve():
                shutil.copy2(exported_onnx, target_onnx)
            if Path(exported_onnx).resolve() != exported_onnx_path.resolve():
                shutil.copy2(exported_onnx, exported_onnx_path)
            print(f"[+] Successfully exported and staged ONNX to {target_onnx} and {exported_onnx_path}")

        # Run validation on test split
        print("[*] Running final validation on test split...")
        test_metrics = best_model.val(data=data_yaml, split="test")
        print("\n" + "=" * 50)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 50)
        print(f"mAP50   : {test_metrics.box.map50:.4f}")
        print(f"mAP50-95: {test_metrics.box.map:.4f}")
        print("=" * 50)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Enhanced PPE YOLO Model")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Backbone architecture (e.g. yolo11s.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device index")
    parser.add_argument("--run-name", type=str, default="yolo_ppe_enhanced", help="Experiment name")
    args = parser.parse_args()

    train_ppe_model(
        data_yaml=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        run_name=args.run_name
    )
