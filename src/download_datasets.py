import os
import sys
import subprocess
import shutil

# Ensure kagglehub is installed
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "pandas"])
    import kagglehub

def download_dataset():
    raw_dir = os.path.join("data", "raw_dataset")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Downloading Auto-rickshaw dataset...")
    # This downloads the dataset to a local cache directory
    path = kagglehub.dataset_download("dataclusterlabs/autorickshaw-image-dataset-niche-vehicle-dataset")
    print(f"Dataset downloaded to cache: {path}")
    
    # We will copy or symlink it to our data folder for easier inspection
    dest = os.path.join(raw_dir, "autorickshaws")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    print(f"Dataset copied to {dest}")
    return dest

if __name__ == "__main__":
    download_dataset()
