import requests
import os

# ── Anchor all paths to the project root (one level above this demo/ folder) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def download_sample():
    raw_videos_dir  = os.path.join(PROJECT_ROOT, 'data', 'raw_videos')
    test_images_dir = os.path.join(PROJECT_ROOT, 'data', 'test_images')
    os.makedirs(raw_videos_dir,  exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    # Sample traffic image (public domain)
    url = "https://github.com/ultralytics/assets/raw/main/traffic.jpg"
    print(f"⬇️  Downloading sample image from {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    out_path = os.path.join(test_images_dir, 'sample_traffic.jpg')
    with open(out_path, 'wb') as f:
        f.write(r.content)
    print(f"✅ Downloaded sample image → {out_path}")

if __name__ == '__main__':
    download_sample()
