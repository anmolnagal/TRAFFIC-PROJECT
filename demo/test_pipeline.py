import sys
import os

# ── Anchor all paths to the project root (one level above this demo/ folder) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
os.chdir(PROJECT_ROOT)  # Make all relative paths inside imported modules work too

import cv2
print("Starting pipeline test...")
print(f"   Project root: {PROJECT_ROOT}")

# Test 1: Preprocessing
print("\n1) Testing preprocessing...")
from preprocessing import preprocess_image

sample_img_path = os.path.join(PROJECT_ROOT, 'data', 'test_images', 'sample_traffic.jpg')
img = cv2.imread(sample_img_path)
if img is not None:
    processed, scale = preprocess_image(img)
    print(f"   Preprocessing OK: {processed.shape}, scale: {scale}")
else:
    print(f"   Sample image not found at: {sample_img_path}")
    print("   Run demo/download_sample_data.py first")

# Test 2: CNN Detection
print("\n2) Testing YOLOv8 detection...")
try:
    from cnn_detector import detect
    bboxes, classes, confs = detect(sample_img_path)
    print(f"   Detection OK: {len(bboxes)} objects")
    print(f"   Classes found: {classes[:3]}{'...' if len(classes)>3 else ''}")
except Exception as e:
    print(f"   Detection failed: {e}")

print("\nPipeline test COMPLETED!")
print("\nNext steps:")
print("   1. python src/data_loader.py --video_dir data/raw_videos")
print("   2. Train: modify cnn_detector.py train_cnn()")
print("   3. Test MATLAB GUI")
