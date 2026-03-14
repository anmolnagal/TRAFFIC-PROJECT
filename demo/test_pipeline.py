import sys
sys.path.insert(0, 'src')

import cv2
print("✅ Starting pipeline test...")

# Test 1: Preprocessing
print("\n1️⃣ Testing preprocessing...")
from preprocessing import preprocess_image
img = cv2.imread('data/test_images/sample_traffic.jpg')
if img is not None:
    processed, scale = preprocess_image(img)
    print(f"   ✅ Preprocessing OK: {processed.shape}, scale: {scale}")
else:
    print("   ❌ Sample image not found - run download_sample_data.py first")

# Test 2: CNN Detection
print("\n2️⃣ Testing YOLOv8 detection...")
try:
    from cnn_detector import detect
    bboxes, classes, confs = detect('data/test_images/sample_traffic.jpg')
    print(f"   ✅ Detection OK: {len(bboxes)} objects")
    print(f"   Classes found: {classes[:3]}{'...' if len(classes)>3 else ''}")
except Exception as e:
    print(f"   ❌ Detection failed: {e}")

print("\n🎉 Pipeline test COMPLETED!")
print("\n🚀 Next steps:")
print("   1. python src/data_loader.py --video_dir data/raw_videos")
print("   2. Train: modify cnn_detector.py train_cnn()")
print("   3. Test MATLAB GUI")
