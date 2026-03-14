import requests
import os

def download_sample():
    os.makedirs('data/raw_videos', exist_ok=True)
    os.makedirs('data/test_images', exist_ok=True)
    
    # Sample traffic image (public domain)
    url = "https://github.com/ultralytics/assets/raw/main/traffic.jpg"
    r = requests.get(url)
    with open('data/test_images/sample_traffic.jpg', 'wb') as f:
        f.write(r.content)
    print("Downloaded sample image")

if __name__ == '__main__':
    download_sample()
