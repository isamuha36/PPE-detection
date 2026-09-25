import shutil
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = Path("data/raw/construction-ppe")
DEST_DIR = Path("data/processed")

# Source classes in construction-ppe:
# 0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none,
# 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots
SRC_TO_TARGET = {
    6: 0,  # Person -> person
    0: 1,  # helmet -> cap_on
    7: 2,  # no_helmet -> cap_off
    2: 5,  # vest -> jacket_on
    1: 7,  # gloves -> gloves_on
    9: 8,  # no_gloves -> gloves_off
}

def remap_and_organize():
    if not SOURCE_DIR.exists():
        print(f"[!] Source dataset directory {SOURCE_DIR} does not exist yet. Please wait for download.")
        return

    splits = ["train", "val", "test"]

    for split in splits:
        src_img_dir = SOURCE_DIR / "images" / split
        src_lbl_dir = SOURCE_DIR / "labels" / split

        dst_img_dir = DEST_DIR / split / "images"
        dst_lbl_dir = DEST_DIR / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            continue

        images = [f for f in src_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"[*] Processing {split} set: {len(images)} images...")

        retained_imgs = 0
        total_boxes = 0

        for img_path in tqdm(images, desc=f"Converting {split}"):
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            new_lines = []

            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        src_cls = int(parts[0])
                        if src_cls in SRC_TO_TARGET:
                            target_cls = SRC_TO_TARGET[src_cls]
                            new_lines.append(f"{target_cls} {' '.join(parts[1:5])}\n")
                            total_boxes += 1

            # Only copy image if it has annotations or for background training
            if new_lines:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                with open(dst_lbl_dir / f"{img_path.stem}.txt", "w") as out:
                    out.writelines(new_lines)
                retained_imgs += 1

        print(f"[+] {split} complete: {retained_imgs} images, {total_boxes} bounding boxes retained.")

    print(f"\n[+] Starter dataset prepared successfully in {DEST_DIR}")

if __name__ == "__main__":
    remap_and_organize()
