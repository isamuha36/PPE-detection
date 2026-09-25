import os
from pathlib import Path
import yaml
import pytest

def test_project_directories():
    base = Path(".")
    expected_dirs = [
        base / "configs",
        base / "data" / "raw",
        base / "data" / "processed",
        base / "data" / "samples",
        base / "models" / "pretrained",
        base / "models" / "trained",
        base / "models" / "exported",
        base / "src" / "dataset",
        base / "src" / "training",
        base / "src" / "optimization",
        base / "src" / "inference",
        base / "src" / "service",
    ]
    for d in expected_dirs:
        assert d.exists(), f"Expected directory {d} does not exist"

def test_data_yaml_config():
    yaml_path = Path("configs/data.yaml")
    assert yaml_path.exists(), "configs/data.yaml is missing"

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "names" in config, "Missing 'names' in data.yaml"
    classes = config["names"]
    expected_classes = [
        "person", "cap_on", "cap_off", "mask_on", "mask_off",
        "jacket_on", "jacket_off", "gloves_on", "gloves_off"
    ]
    for c in expected_classes:
        assert c in classes.values(), f"Class '{c}' not found in data.yaml"
