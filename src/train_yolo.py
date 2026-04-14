"""
train_yolo.py
=============
Fine-tunes YOLOv8n on the merged Indian Vehicles + Congestion dataset.

Run from project root AFTER running src/prepare_dataset.py:
    python src/train_yolo.py

Output:
    models/indian_vehicles_yolo.pt   ← best weights copied here
    runs/train/indian_vehicles/      ← full training artefacts
"""

import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import torch
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_YAML    = ROOT / "configs" / "indian_vehicles.yaml"
BASE_MODEL   = ROOT / "yolov8n.pt"
OUTPUT_MODEL = ROOT / "models" / "indian_vehicles_yolo.pt"

# ── Training hyper-params ─────────────────────────────────────────────────────
EPOCHS   = 80          # more data → more epochs; early-stop handles over-training
IMG_SIZE = 640
BATCH    = 4           # CPU-safe; small batch is fine for our dataset size
PATIENCE = 25          # early-stop patience
WORKERS  = 0           # Windows + Intel Iris Xe: must be 0

# Intel Iris Xe has no CUDA → force CPU
DEVICE = "cpu"


def check_data() -> int:
    if not DATA_YAML.exists():
        sys.exit(
            "❌  configs/indian_vehicles.yaml not found!\n"
            "    Run  python src/prepare_dataset.py  first."
        )

    import yaml
    with open(DATA_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg.get("path", str(ROOT / "data")))
    train_rel = cfg.get("train", "train/images")
    train_dir = (data_path / train_rel) if not Path(train_rel).is_absolute() else Path(train_rel)

    if not train_dir.is_dir():
        sys.exit(
            f"❌  Training images not found at {train_dir}\n"
            "    Run  python src/prepare_dataset.py  first."
        )

    imgs = list(train_dir.glob("*"))
    imgs = [f for f in imgs if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    print(f"✅  Found {len(imgs)} training images in {train_dir}")
    return len(imgs)


def train():
    n_imgs = check_data()

    print(f"\n🚀  Starting YOLOv8n fine-tune")
    print(f"    Device   : CPU (Intel Iris Xe — CUDA not available)")
    print(f"    Images   : {n_imgs}  training")
    print(f"    Epochs   : {EPOCHS}  (early-stop patience={PATIENCE})")
    print(f"    Batch    : {BATCH}")
    print(f"    Config   : {DATA_YAML}")
    print(f"    Base     : {BASE_MODEL}\n")

    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data     = str(DATA_YAML),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        patience = PATIENCE,
        workers  = WORKERS,
        project  = "runs/train",
        name     = "indian_vehicles",
        exist_ok = True,
        plots    = True,
        verbose  = True,

        # ── Augmentation ──────────────────────────────────────────────────────
        # Colour / lighting jitter
        hsv_h       = 0.020,
        hsv_s       = 0.70,
        hsv_v       = 0.40,

        # Geometry (simulates different camera angles, elevations)
        degrees     = 10.0,
        translate   = 0.15,
        scale       = 0.5,
        shear       = 2.0,
        perspective = 0.0005,
        flipud      = 0.0,
        fliplr      = 0.5,

        # Scene-level augmentation
        mosaic      = 1.0,
        mixup       = 0.10,
        copy_paste  = 0.20,

        # Schedule
        warmup_epochs = 3,
        close_mosaic  = 10,
        lr0           = 0.01,
        lrf           = 0.01,
        weight_decay  = 0.0005,
    )

    # ── Copy best weights to models/ ──────────────────────────────────────────
    run_dir = Path("runs/train/indian_vehicles")
    candidates = [
        run_dir / "weights" / "best.pt",
        ROOT / "runs" / "train" / "indian_vehicles" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "train"  / "weights" / "best.pt",
    ]
    best_weights = next((p for p in candidates if p.exists()), None)

    if best_weights:
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, OUTPUT_MODEL)
        print(f"\n✅  Best model saved → {OUTPUT_MODEL}")
        print(f"    (copied from {best_weights})")
    else:
        print("⚠️  Could not find best.pt — check runs/train/indian_vehicles/weights/")

    # ── Print metrics ──────────────────────────────────────────────────────────
    try:
        metrics = results.results_dict
        print(f"\n📊  Final Metrics:")
        print(f"    mAP50    : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"    mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"    Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"    Recall   : {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass

    print("\n🎉  Training complete!  Run:  python demo/app.py")


if __name__ == "__main__":
    train()
