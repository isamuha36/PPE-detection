import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Class mapping helper to unify external public PPE datasets into our 9-class standard
TARGET_CLASSES = {
    0: "person",
    1: "cap_on",
    2: "cap_off",
    3: "mask_on",
    4: "mask_off",
    5: "jacket_on",
    6: "jacket_off",
    7: "gloves_on",
    8: "gloves_off"
}

NAME_TO_TARGET_ID = {v: k for k, v in TARGET_CLASSES.items()}

# Common synonyms in public PPE datasets (e.g. Roboflow, SHWD, Kaggle)
SYNONYMS = {
    "hard-hat": "cap_on",
    "hardhat": "cap_on",
    "helmet": "cap_on",
    "safety-helmet": "cap_on",
    "head": "cap_off",
    "no-helmet": "cap_off",
    "no-hardhat": "cap_off",
    "no_helmet": "cap_off",

    "mask": "mask_on",
    "face-mask": "mask_on",
    "respirator": "mask_on",
    "no-mask": "mask_off",
    "no_mask": "mask_off",

    "vest": "jacket_on",
    "safety-vest": "jacket_on",
    "hi-vis": "jacket_on",
    "jacket": "jacket_on",
    "no-vest": "jacket_off",
    "no_vest": "jacket_off",

    "gloves": "gloves_on",
    "glove": "gloves_on",
    "no-gloves": "gloves_off",
    "no_gloves": "gloves_off",

    "person": "person",
    "worker": "person"
}

def map_label(source_name: str) -> int:
    """Maps a source dataset class label name to target class ID."""
    clean_name = source_name.lower().strip()
    target_name = SYNONYMS.get(clean_name, None)
    if target_name and target_name in NAME_TO_TARGET_ID:
        return NAME_TO_TARGET_ID[target_name]
    return -1

def convert_annotation_file(
    src_label_file: Path,
    dst_label_file: Path,
    source_class_names: List[str]
) -> int:
    """
    Reads a YOLO label file from a foreign dataset, remaps class IDs to standard,
    and writes to target label file. Returns number of retained bounding boxes.
    """
    retained_boxes = 0
    new_lines = []

    if not src_label_file.exists():
        return 0

    with open(src_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_cls_id = int(parts[0])
            if orig_cls_id < len(source_class_names):
                orig_name = source_class_names[orig_cls_id]
                new_cls_id = map_label(orig_name)
                if new_cls_id != -1:
                    new_lines.append(f"{new_cls_id} {' '.join(parts[1:5])}\n")
                    retained_boxes += 1

    if new_lines:
        dst_label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label_file, "w") as out:
            out.writelines(new_lines)

    return retained_boxes
