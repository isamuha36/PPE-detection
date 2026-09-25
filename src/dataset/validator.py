import os
from pathlib import Path
from typing import Dict
import yaml

def validate_dataset(data_yaml_path: str) -> Dict[str, any]:
    """
    Validates YOLO dataset paths, checks for missing files, and summarizes class statistics.
    """
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(data_yaml_path).parent / config.get("path", "")
    classes = config.get("names", {})
    splits = {"train": config.get("train"), "val": config.get("val"), "test": config.get("test")}

    stats = {"splits": {}, "total_images": 0, "class_counts": {c: 0 for c in classes.values()}}

    for split_name, rel_path in splits.items():
        if not rel_path:
            continue
        img_dir = (base_dir / rel_path).resolve()
        lbl_dir = img_dir.parent / "labels"
        if not img_dir.exists():
            stats["splits"][split_name] = {"found": False, "image_count": 0}
            continue

        images = [f for f in img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        stats["splits"][split_name] = {"found": True, "image_count": len(images)}
        stats["total_images"] += len(images)

        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                with open(lf, "r", encoding="utf-8") as lf_in:
                    for line in lf_in:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id in classes:
                                c_name = classes[cls_id]
                                stats["class_counts"][c_name] += 1

    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    results = validate_dataset(args.config)
    print("Dataset Validation Summary:")
    print(results)
