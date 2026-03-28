"""
diagnose_model.py
=================
Quick diagnostic to see what the model actually predicts on an image.
Run: python src/diagnose_model.py

Pass an image path as argument, or it will find the first image in data/train/images/
"""
import os, sys
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")

# Find a test image
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    train_dir = os.path.join(ROOT, "data", "train", "images")
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [f for f in os.listdir(train_dir) if os.path.splitext(f)[1].lower() in exts]
    if not imgs:
        print("No images found in data/train/images/"); sys.exit(1)
    img_path = os.path.join(train_dir, imgs[0])

print(f"\n{'='*55}")
print(f"  Model:  {MODEL_PATH}")
print(f"  Exists: {os.path.exists(MODEL_PATH)}")
print(f"  Image:  {img_path}")
print(f"{'='*55}\n")

# Load model
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f"Model class names: {model.names}\n")

# Run with essentially no confidence filter
results = model(img_path, conf=0.01, verbose=True)[0]

print(f"\n{'='*55}")
print(f"  Raw detections (conf >= 0.01):")
print(f"{'='*55}")

if results.boxes is None or len(results.boxes) == 0:
    print("  ❌  ZERO detections even at conf=0.01")
    print("  → The model is not producing any predictions at all.")
    print("  → This usually means the weights file is from a failed/incomplete training run.")
else:
    print(f"  ✅  {len(results.boxes)} detection(s) found:")
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        name = model.names.get(cls_id, f"cls_{cls_id}")
        print(f"    [{i+1}] {name:<15} conf={conf:.4f}  box=({x1},{y1},{x2},{y2})")

    # Show by confidence threshold what survives
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        n = sum(1 for b in results.boxes if float(b.conf[0]) >= thresh)
        print(f"\n  At conf >= {thresh:.2f}: {n} detection(s) survive")

# Also save an annotated image so we can see visually
out_path = os.path.join(ROOT, "_diagnose_output.jpg")
annotated = results.plot()
cv2.imwrite(out_path, annotated)
print(f"\n  Annotated image saved to: {out_path}")
print(f"  (Open this to see what the model detected visually)\n")
