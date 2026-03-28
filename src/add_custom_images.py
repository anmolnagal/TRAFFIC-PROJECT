"""
add_custom_images.py
====================
Step 1:  Drop your congestion images into  data/custom_congestion/images/
Step 2:  Run this script from project root:
             python src/add_custom_images.py
Step 3:  Script auto-labels every image with your YOLO model.
Step 4:  Review / correct labels in LabelImg (optional but recommended).
Step 5:  Script merges images + labels into  data/train/  and  data/valid/
Step 6:  Re-run  python src/train_yolo.py  to fine-tune on the expanded set.

Folder layout expected:
    data/
    ├── train/images/        ← existing dataset
    ├── train/labels/
    ├── valid/images/
    ├── valid/labels/
    └── custom_congestion/
        ├── images/          ← PUT YOUR IMAGES HERE
        └── labels/          ← auto-generated here, then review before merging
"""

import os
import sys
import shutil
import random
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

CUSTOM_DIR   = os.path.join(ROOT, "data", "custom_congestion")
IMG_IN_DIR   = os.path.join(CUSTOM_DIR, "images")
LBL_IN_DIR   = os.path.join(CUSTOM_DIR, "labels")
TRAIN_IMG    = os.path.join(ROOT, "data", "train", "images")
TRAIN_LBL    = os.path.join(ROOT, "data", "train", "labels")
VALID_IMG    = os.path.join(ROOT, "data", "valid", "images")
VALID_LBL    = os.path.join(ROOT, "data", "valid", "labels")

CUSTOM_MODEL   = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")
FALLBACK_MODEL = os.path.join(ROOT, "yolov8n.pt")

CONF_THRESH = 0.30   # lower = more boxes (good for congestion; you'll prune bad ones manually)
VAL_SPLIT   = 0.15   # 15% of custom images go to validation

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# COCO class ID → our label name (used if falling back to yolov8n.pt)
COCO_MAP = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Our dataset class names (must match indian_vehicles.yaml)
CLASS_NAMES = ["auto", "bicycle", "bus", "car", "motorcycle",
               "pedestrian", "tempo", "tractor", "truck", "van"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _class_id(name: str) -> int:
    """Return class index. Unknown names → -1 (will be skipped)."""
    name = name.lower().strip()
    if name in CLASS_NAMES:
        return CLASS_NAMES.index(name)
    # aliases
    alias = {
        "motorbike": "motorcycle", "two_wheeler": "motorcycle",
        "auto-rickshaw": "auto",   "auto rickshaw": "auto",
        "rickshaw": "auto",        "tuk-tuk": "auto",
        "minibus": "bus",          "microbus": "van",
    }
    mapped = alias.get(name)
    if mapped and mapped in CLASS_NAMES:
        return CLASS_NAMES.index(mapped)
    return -1


def _load_model():
    from ultralytics import YOLO
    if os.path.exists(CUSTOM_MODEL):
        print(f"  ✅  Using custom model:   {CUSTOM_MODEL}")
        return YOLO(CUSTOM_MODEL), True
    print(f"  ⚠️   Custom model not found. Using COCO fallback: {FALLBACK_MODEL}")
    return YOLO(FALLBACK_MODEL), False


def _yolo_label_line(cls_id, x1, y1, x2, y2, img_w, img_h) -> str:
    """Convert pixel bbox to YOLO normalised format."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    # clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ── Stage 1: auto-label ───────────────────────────────────────────────────────

def auto_label(skip_existing: bool = True):
    os.makedirs(IMG_IN_DIR, exist_ok=True)
    os.makedirs(LBL_IN_DIR, exist_ok=True)

    images = [f for f in os.listdir(IMG_IN_DIR)
              if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
    if not images:
        print(f"\n❌  No images found in  {IMG_IN_DIR}")
        print("    Copy your congestion images there and re-run.")
        sys.exit(1)

    print(f"\n🔍  Found {len(images)} image(s) to label in  {IMG_IN_DIR}")
    model, is_custom = _load_model()

    labeled = 0
    skipped = 0
    total_boxes = 0

    for fname in images:
        img_path = os.path.join(IMG_IN_DIR, fname)
        lbl_path = os.path.join(LBL_IN_DIR,
                                os.path.splitext(fname)[0] + ".txt")

        if skip_existing and os.path.exists(lbl_path):
            skipped += 1
            continue

        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️   Cannot read {fname} — skipping.")
            continue
        h, w = img.shape[:2]

        results = model(img_path, conf=CONF_THRESH, verbose=False)[0]
        lines = []

        for box in results.boxes:
            cls_id_raw = int(box.cls[0])
            conf       = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if is_custom:
                raw_name = results.names.get(cls_id_raw, "")
                final_id = _class_id(raw_name)
            else:
                if cls_id_raw not in COCO_MAP:
                    continue
                final_id = _class_id(COCO_MAP[cls_id_raw])

            if final_id < 0:
                continue

            lines.append(_yolo_label_line(final_id, x1, y1, x2, y2, w, h))

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        total_boxes += len(lines)
        labeled += 1
        print(f"  [{labeled:>3}/{len(images)}]  {fname:<40}  {len(lines):>3} box(es)  conf≥{CONF_THRESH}")

    print(f"\n✅  Auto-labeled {labeled} image(s)  ({skipped} skipped — already had labels)")
    print(f"    Total boxes written: {total_boxes}")
    print(f"\n⚠️   REVIEW STEP: Visually check the labels before merging!")
    print(f"    Quick check command (if LabelImg installed):")
    print(f"      labelImg \"{IMG_IN_DIR}\" \"{LBL_IN_DIR}\"")


# ── Stage 2: merge into train/valid ──────────────────────────────────────────

def merge(dry_run: bool = False):
    """Copy labeled custom images into data/train and data/valid."""
    images = [f for f in os.listdir(IMG_IN_DIR)
              if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
    labels = {os.path.splitext(f)[0] for f in os.listdir(LBL_IN_DIR)
              if f.endswith(".txt")}

    # Only merge images that have a label file
    paired = [f for f in images if os.path.splitext(f)[0] in labels]
    unlabeled = [f for f in images if os.path.splitext(f)[0] not in labels]

    if unlabeled:
        print(f"\n⚠️   {len(unlabeled)} image(s) have no label file — will NOT be merged:")
        for f in unlabeled:
            print(f"      {f}")

    if not paired:
        print("\n❌  No paired image+label files found. Run auto_label() first.")
        sys.exit(1)

    random.shuffle(paired)
    n_val = max(1, int(len(paired) * VAL_SPLIT))
    val_set   = set(paired[:n_val])
    train_set = set(paired[n_val:])

    print(f"\n📦  Merging {len(train_set)} image(s) → train,  {len(val_set)} → valid")

    os.makedirs(TRAIN_IMG, exist_ok=True); os.makedirs(TRAIN_LBL, exist_ok=True)
    os.makedirs(VALID_IMG, exist_ok=True); os.makedirs(VALID_LBL, exist_ok=True)

    def _copy(fname, dst_img, dst_lbl):
        base  = os.path.splitext(fname)[0]
        src_i = os.path.join(IMG_IN_DIR, fname)
        src_l = os.path.join(LBL_IN_DIR, base + ".txt")
        if not dry_run:
            shutil.copy2(src_i, os.path.join(dst_img, fname))
            shutil.copy2(src_l, os.path.join(dst_lbl, base + ".txt"))

    for f in train_set:
        _copy(f, TRAIN_IMG, TRAIN_LBL)
        print(f"  [train]  {f}")
    for f in val_set:
        _copy(f, VALID_IMG, VALID_LBL)
        print(f"  [valid]  {f}")

    if dry_run:
        print("\n(DRY RUN — no files copied)")
    else:
        # Count totals
        n_train_total = len([x for x in os.listdir(TRAIN_IMG)
                             if os.path.splitext(x)[1].lower() in SUPPORTED_EXT])
        n_valid_total = len([x for x in os.listdir(VALID_IMG)
                             if os.path.splitext(x)[1].lower() in SUPPORTED_EXT])
        print(f"\n✅  Merge complete!")
        print(f"    data/train/images  → {n_train_total} total images")
        print(f"    data/valid/images  → {n_valid_total} total images")
        print(f"\n🚀  Now run:  python src/train_yolo.py")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label & merge custom congestion images into training dataset.")
    parser.add_argument("--label-only",  action="store_true",
                        help="Only run auto-labeling, skip merge step.")
    parser.add_argument("--merge-only",  action="store_true",
                        help="Skip labeling, just merge already-labeled images.")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Preview merge without copying any files.")
    parser.add_argument("--relabel",     action="store_true",
                        help="Re-label even images that already have a label file.")
    args = parser.parse_args()

    print("=" * 60)
    print("  TRAFFIC PROJECT — Custom Image Labeler & Dataset Merger")
    print("=" * 60)

    if not args.merge_only:
        auto_label(skip_existing=not args.relabel)

    if not args.label_only:
        print()
        ans = input("Have you reviewed/corrected the labels? Merge now? [y/N] ").strip().lower()
        if ans == "y":
            merge(dry_run=args.dry_run)
        else:
            print("\n⏸️   Merge skipped. Re-run with  --merge-only  when ready.")


if __name__ == "__main__":
    main()
