"""
train_yolo.py
=============
Fine-tunes YOLOv8n on the Indian Vehicles dataset.

Run from project root:
  python src/train_yolo.py

Output:
  models/indian_vehicles_yolo.pt   ← best weights (copy here after training)
  runs/train/indian_vehicles/      ← full training run artefacts
"""

import os
import sys
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

import torch
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_YAML   = os.path.join(ROOT, "configs", "indian_vehicles.yaml")
BASE_MODEL  = os.path.join(ROOT, "yolov8n.pt")
OUTPUT_MODEL = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")

# ── Training hyper-params ────────────────────────────────────────────────────
EPOCHS    = 60         # enough for 26-image dataset; early-stop will kick in
IMG_SIZE  = 640
BATCH     = 4          # small dataset → small batch; avoids memory issues on Intel Xe
PATIENCE  = 20         # wait longer before early-stopping (small dataset oscillates)
WORKERS   = 0         # Intel Xe / Windows: must be 0

# ── Device ───────────────────────────────────────────────────────────────────
# Intel Iris Xe is an integrated GPU with no CUDA support.
# PyTorch will use CPU — still fast enough for 26-60 images.
DEVICE = "cpu"


def check_data():
    if not os.path.exists(DATA_YAML):
        sys.exit(
            "❌  configs/indian_vehicles.yaml not found!\n"
            "    Run  python src/download_datasets.py  first."
        )
    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("path", "data")
    train_rel = cfg.get("train", "train/images")
    train_dir = os.path.join(data_path, train_rel) if not os.path.isabs(train_rel) else train_rel
    if not os.path.isdir(train_dir):
        sys.exit(
            f"❌  Training images not found at {train_dir}\n"
            "    Run  python src/download_datasets.py  first."
        )
    imgs = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"✅  Found {len(imgs)} training images in {train_dir}")
    return len(imgs)


def train():
    n_imgs = check_data()
    device_label = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE == "0" else "CPU"
    print(f"\n🚀  Starting YOLOv8n fine-tune")
    print(f"    Device  : {device_label}")
    print(f"    Images  : {n_imgs}")
    print(f"    Epochs  : {EPOCHS}  (patience={PATIENCE})")
    print(f"    Config  : {DATA_YAML}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        patience  = PATIENCE,
        workers   = WORKERS,
        project   = "runs/train",
        name      = "indian_vehicles",
        exist_ok  = True,
        plots     = True,
        verbose   = True,

        # ── Augmentation — heavy to compensate for small dataset ──────────────
        # Colour / lighting (handles dusk, night, mixed artificial light)
        hsv_h     = 0.020,   # hue shift
        hsv_s     = 0.70,    # saturation jitter
        hsv_v     = 0.40,    # brightness jitter

        # Geometry (handles elevated camera angle, different perspectives)
        degrees   = 10.0,    # rotation ± 10°
        translate = 0.15,    # shift ± 15 %
        scale     = 0.6,     # zoom ± 60 %
        shear     = 2.0,     # slight shear
        perspective = 0.0005, # perspective warp (simulates different angles)
        flipud    = 0.0,     # don't flip upside-down (vehicles should be right-way up)
        fliplr    = 0.5,     # horizontal flip — fine for traffic

        # Scene-level augmentation (critical for congestion simulation)
        mosaic    = 1.0,     # always use mosaic (combines 4 images → dense scene)
        mixup     = 0.15,    # slightly blend two images
        copy_paste = 0.30,   # copy vehicle instances and paste into other images
                             # ↑ artificially increases vehicle density in training

        # Training schedule
        warmup_epochs = 5,   # warm up LR for first 5 epochs
        close_mosaic  = 10,  # turn off mosaic for last 10 epochs for stability
        lr0       = 0.01,
        lrf       = 0.01,
        weight_decay = 0.0005,
    )

    # YOLO may save to runs/train/ or runs/detect/train/ depending on version
    # Try both locations
    candidates = [
        os.path.join(ROOT, "runs", "train",  "indian_vehicles", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "detect", "runs", "train", "indian_vehicles", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "detect", "train2", "weights", "best.pt"),
    ]
    best_weights = next((p for p in candidates if os.path.exists(p)), None)

    if best_weights:
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        shutil.copy2(best_weights, OUTPUT_MODEL)
        print(f"\n✅  Best model saved to: {OUTPUT_MODEL}")
        print(f"    (copied from: {best_weights})")
    else:
        print(f"⚠️   Could not find best.pt — check runs/train/indian_vehicles/weights/")

    # Print summary metrics
    try:
        metrics = results.results_dict
        print(f"\n📊  Final Metrics:")
        print(f"    mAP50   : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"    mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"    Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"    Recall   : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    train()
