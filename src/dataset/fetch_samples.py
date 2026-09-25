import urllib.request
from pathlib import Path

# Sample open-access PPE testing images
SAMPLE_URLS = {
    "sample_worker_1.jpg": "https://images.unsplash.com/photo-1504307651254-35680f356dfd?q=80&w=640&auto=format&fit=crop", # Construction worker with helmet and vest
    "sample_worker_2.jpg": "https://images.unsplash.com/photo-1541888946425-d0fbb186f5f7?q=80&w=640&auto=format&fit=crop", # Worker on site
}

def download_samples(target_dir: str = "data/samples"):
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_URLS.items():
        file_path = dest / filename
        if not file_path.exists():
            print(f"[*] Downloading sample {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"[+] Saved to {file_path}")
            except Exception as e:
                print(f"[!] Failed to download {filename}: {e}")
        else:
            print(f"[*] Sample {filename} already exists.")

if __name__ == "__main__":
    download_samples()
