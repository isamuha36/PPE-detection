import urllib.request
import zipfile
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "construction-ppe.zip"
    extract_folder = RAW_DIR / "construction-ppe"

    if not zip_path.exists():
        print(f"[*] Downloading starter PPE dataset from {URL}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="construction-ppe.zip") as t:
            urllib.request.urlretrieve(URL, filename=zip_path, reporthook=t.update_to)
        print("[+] Download complete.")
    else:
        print("[*] Archive construction-ppe.zip already exists.")

    if not extract_folder.exists():
        print(f"[*] Extracting {zip_path} to {extract_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("[+] Extraction complete.")
    else:
        print("[*] Extracted directory already exists.")

    print(f"[*] Starter dataset ready at: {extract_folder}")

if __name__ == "__main__":
    download_and_extract()
