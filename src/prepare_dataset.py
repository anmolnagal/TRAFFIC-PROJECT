"""
prepare_dataset.py
==================
Converts Pascal VOC XML annotations (Indian_vehicle_dataset/) to YOLO format
and merges them with the existing custom_congestion dataset into clean
train/ and valid/ splits.

Run from project root:
    python src/prepare_dataset.py
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data"

# ── Unified class list (10 classes, matches indian_vehicles.yaml) ─────────────
CLASSES = [
    "auto",         # 0  (auto-rickshaw / tuk-tuk)
    "bicycle",      # 1
    "bus",          # 2
    "car",          # 3
    "motorcycle",   # 4
    "pedestrian",   # 5
    "tempo",        # 6  (commercial tempo / mini-truck)
    "tractor",      # 7
    "truck",        # 8
    "van",          # 9
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# ── Map raw dataset class names → unified names ────────────────────────────────
NAME_MAP = {
    # Indian_vehicle_dataset XML variants
    "auto":            "auto",
    "autorickshaw":    "auto",
    "auto_rickshaw":   "auto",
    "e_autorickshaw":  "auto",
    "e_rickshaw":      "auto",
    "bicycle":         "bicycle",
    "cycle":           "bicycle",
    "bus":             "bus",
    "electric_bus":    "bus",
    "car":             "car",
    "motorcycle":      "motorcycle",
    "two_wheelers":    "motorcycle",
    "two_wheeler":     "motorcycle",
    "pedestrian":      "pedestrian",
    "tempo":           "tempo",
    "tractor":         "tractor",
    "truck":           "truck",
    "vehicle_truck":   "truck",
    "van":             "van",
    # congestion labels — existing YOLO labels already use class IDs;
    # we handle those separately below
}

VALID_SPLIT = 0.20   # 20 % of each source goes to validation
random.seed(42)

# ── Output dirs ───────────────────────────────────────────────────────────────
TRAIN_IMGS   = DATA / "train" / "images"
TRAIN_LABELS = DATA / "train" / "labels"
VALID_IMGS   = DATA / "valid" / "images"
VALID_LABELS = DATA / "valid" / "labels"

for d in [TRAIN_IMGS, TRAIN_LABELS, VALID_IMGS, VALID_LABELS]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise(name: str) -> str | None:
    """Map a raw class name to unified name, or None to skip."""
    return NAME_MAP.get(name.strip().lower().replace(" ", "_"))


def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse a Pascal VOC XML file and return YOLO-format lines.
    Skips objects whose class is not in NAME_MAP.
    """
    lines = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try reading image size from XML if not provided
        size = root.find("size")
        if size is not None:
            try:
                img_w = int(float(size.findtext("width") or img_w))
                img_h = int(float(size.findtext("height") or img_h))
            except ValueError:
                pass

        for obj in root.findall("object"):
            raw_name = (obj.findtext("name") or "").strip()
            unified  = normalise(raw_name)
            if unified is None:
                print(f"  ⚠  Skipping unknown class '{raw_name}' in {xml_path.name}")
                continue

            cls_id = CLASS_TO_ID[unified]
            bbox   = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            if img_w <= 0 or img_h <= 0:
                continue

            x_c = ((xmin + xmax) / 2) / img_w
            y_c = ((ymin + ymax) / 2) / img_h
            w   = (xmax - xmin) / img_w
            h   = (ymax - ymin) / img_h

            # Clamp to [0, 1]
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w   = max(0.0, min(1.0, w))
            h   = max(0.0, min(1.0, h))

            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    except ET.ParseError as e:
        print(f"  ✗ XML parse error in {xml_path.name}: {e}")
    return lines


def copy_pair(img_src: Path, lbl_content: str, is_val: bool, stem: str | None = None):
    """Copy image + write label to the correct split directory."""
    stem = stem or img_src.stem
    suffix = img_src.suffix
    img_dst  = (VALID_IMGS   if is_val else TRAIN_IMGS)   / (stem + suffix)
    lbl_dst  = (VALID_LABELS if is_val else TRAIN_LABELS) / (stem + ".txt")

    shutil.copy2(img_src, img_dst)
    lbl_dst.write_text(lbl_content, encoding="utf-8")


# ── Step 1: Process Indian_vehicle_dataset (XML → YOLO) ──────────────────────

print("\n" + "="*60)
print("  Step 1: Converting Indian_vehicle_dataset XML → YOLO")
print("="*60)

iv_dir = DATA / "Indian_vehicle_dataset"
img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

xml_files = sorted(iv_dir.glob("*.xml"))
img_files = {f.stem: f for f in iv_dir.iterdir() if f.suffix.lower() in img_exts}

converted, skipped = 0, 0
iv_pairs = []   # (img_path, yolo_label_str)

for xml_path in xml_files:
    stem = xml_path.stem
    img_path = img_files.get(stem)
    if img_path is None:
        print(f"  ⚠  No image for {xml_path.name}, skipping")
        skipped += 1
        continue

    lines = voc_xml_to_yolo(xml_path, img_w=0, img_h=0)
    if not lines:
        print(f"  ⚠  No valid annotations in {xml_path.name}, skipping")
        skipped += 1
        continue

    iv_pairs.append((img_path, "\n".join(lines)))
    converted += 1

print(f"\n  ✅  Converted {converted} images  |  Skipped {skipped}")

# Split 80/20
random.shuffle(iv_pairs)
n_val = max(1, int(len(iv_pairs) * VALID_SPLIT))
iv_val   = iv_pairs[:n_val]
iv_train = iv_pairs[n_val:]

for img_path, label in iv_train:
    copy_pair(img_path, label, is_val=False, stem="iv_" + img_path.stem[:40])
for img_path, label in iv_val:
    copy_pair(img_path, label, is_val=True,  stem="iv_" + img_path.stem[:40])

print(f"  → Train: {len(iv_train)}  |  Val: {len(iv_val)}")


# ── Step 2: Process custom_congestion dataset ─────────────────────────────────

print("\n" + "="*60)
print("  Step 2: Merging custom_congestion dataset")
print("="*60)

cc_imgs   = DATA / "custom_congestion" / "images"
cc_labels = DATA / "custom_congestion" / "labels"

cc_pairs = []
for img_path in sorted(cc_imgs.iterdir()):
    if img_path.suffix.lower() not in img_exts:
        continue
    lbl_path = cc_labels / (img_path.stem + ".txt")
    if not lbl_path.exists():
        print(f"  ⚠  No label for {img_path.name}, skipping")
        continue
    cc_pairs.append((img_path, lbl_path.read_text(encoding="utf-8")))

random.shuffle(cc_pairs)
n_val_cc = max(1, int(len(cc_pairs) * VALID_SPLIT))
cc_val   = cc_pairs[:n_val_cc]
cc_train = cc_pairs[n_val_cc:]

for img_path, label in cc_train:
    copy_pair(img_path, label, is_val=False, stem="cc_" + img_path.stem[:40])
for img_path, label in cc_val:
    copy_pair(img_path, label, is_val=True,  stem="cc_" + img_path.stem[:40])

print(f"  ✅  Congestion  →  Train: {len(cc_train)}  |  Val: {len(cc_val)}")


# ── Step 3: Remove stale cache files ─────────────────────────────────────────

print("\n" + "="*60)
print("  Step 3: Removing stale cache files")
print("="*60)

for cache in DATA.rglob("*.cache"):
    cache.unlink()
    print(f"  🗑  Removed {cache}")


# ── Step 4: Print dataset summary ─────────────────────────────────────────────

print("\n" + "="*60)
print("  Step 4: Dataset Summary")
print("="*60)

def count_classes(label_dir: Path) -> Counter:
    c = Counter()
    for lf in label_dir.glob("*.txt"):
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(CLASSES):
                        c[CLASSES[cls_id]] += 1
                except ValueError:
                    pass
    return c

train_counts = count_classes(TRAIN_LABELS)
val_counts   = count_classes(VALID_LABELS)

train_img_count = len(list(TRAIN_IMGS.glob("*")))
val_img_count   = len(list(VALID_IMGS.glob("*")))

print(f"\n  Total train images : {train_img_count}")
print(f"  Total val   images : {val_img_count}")
print(f"\n  {'Class':<15} {'Train':>8} {'Val':>8}")
print(f"  {'-'*33}")
for cls in CLASSES:
    print(f"  {cls:<15} {train_counts.get(cls, 0):>8} {val_counts.get(cls, 0):>8}")
total_train = sum(train_counts.values())
total_val   = sum(val_counts.values())
print(f"  {'-'*33}")
print(f"  {'TOTAL':<15} {total_train:>8} {total_val:>8}")
print()


# ── Step 5: Update configs/indian_vehicles.yaml ──────────────────────────────

print("="*60)
print("  Step 5: Updating configs/indian_vehicles.yaml")
print("="*60)

yaml_content = f"""# Indian Vehicles + Congestion — unified dataset config
# Auto-generated by src/prepare_dataset.py

path: {str(DATA).replace(chr(92), '/')}
train: train/images
val: valid/images

nc: {len(CLASSES)}
names:
"""
for i, cls in enumerate(CLASSES):
    yaml_content += f"  {i}: {cls}\n"

yaml_path = ROOT / "configs" / "indian_vehicles.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"  ✅  Written to {yaml_path}")

print("\n" + "="*60)
print("  ✅  Dataset preparation complete!")
print("  Next step:  python src/train_yolo.py")
print("="*60 + "\n")
