"""
download_datasets.py
====================
Downloads the Indian Vehicles dataset (YOLO format) from Roboflow Universe.

Dataset:  Indian Vehicles Detection
Source:   https://universe.roboflow.com/indian-vehicles-obhqj/indian-vehicles-qfkn2
Classes:  auto, bicycle, bus, car, motorcycle, pedestrian, tempo, tractor, truck, van

How to get your FREE Roboflow API key:
  1. Visit https://roboflow.com  →  Sign up (free, no credit card)
  2. Settings → API  →  Copy your Private API Key
  3. Set env var:   set ROBOFLOW_API_KEY=your_key_here
     OR let this script prompt you.

Run from project root:
  python src/download_datasets.py
"""

import os
import sys
import shutil
import subprocess
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
CFG_DIR  = os.path.join(ROOT, "configs")

# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "auto",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "motorcycle",
    5: "pedestrian",
    6: "tempo",
    7: "tractor",
    8: "truck",
    9: "van",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_pkg(package: str):
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"Installing {package}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def _get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        print(f"✅ Using ROBOFLOW_API_KEY from environment.")
        return key
    print("\n" + "="*60)
    print("  Roboflow API key not found in environment.")
    print("  Get a FREE key at: https://roboflow.com → Settings → API")
    print("="*60)
    key = input("  Paste your Roboflow API key here: ").strip()
    if not key:
        sys.exit("❌  No API key provided. Exiting.")
    # Optionally persist for this session
    os.environ["ROBOFLOW_API_KEY"] = key
    return key


# ── primary download via Roboflow ─────────────────────────────────────────────

def download_roboflow(api_key: str) -> str:
    """Download Indian Vehicles dataset from Roboflow Universe."""
    _ensure_pkg("roboflow")
    from roboflow import Roboflow  # type: ignore

    rf = Roboflow(api_key=api_key)

    # Public dataset on Roboflow Universe – Indian Vehicles Detection
    # URL: https://universe.roboflow.com/indian-vehicles-obhqj/indian-vehicles-qfkn2
    workspace = "indian-vehicles-obhqj"
    project_name = "indian-vehicles-qfkn2"
    version_num = 3   # latest stable version

    print(f"\n📥  Downloading from Roboflow Universe …")
    print(f"    workspace : {workspace}")
    print(f"    project   : {project_name}  v{version_num}")

    try:
        project  = rf.workspace(workspace).project(project_name)
        dataset  = project.version(version_num).download("yolov8", location=DATA_DIR, overwrite=True)
        print(f"✅  Dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"⚠️  Primary dataset failed: {e}")
        print("    Trying alternate Indian vehicle dataset…")
        # Alternate: IDD-style Indian traffic dataset
        try:
            project  = rf.workspace("university-bf58j").project("idd-obhqj")
            dataset  = project.version(1).download("yolov8", location=DATA_DIR, overwrite=True)
            print(f"✅  Alternate dataset downloaded to: {dataset.location}")
            return dataset.location
        except Exception as e2:
            print(f"⚠️  Alternate also failed: {e2}")
            return _fallback_kaggle()


# ── fallback: Kaggle (autorickshaw dataset) ───────────────────────────────────

def _fallback_kaggle() -> str:
    """Fall back to the confirmed-working Kaggle autorickshaw dataset."""
    print("\n📥  Falling back to Kaggle autorickshaw dataset…")
    _ensure_pkg("kagglehub")
    import kagglehub  # type: ignore
    import xml.etree.ElementTree as ET
    import random

    raw_path = kagglehub.dataset_download(
        "dataclusterlabs/autorickshaw-image-dataset-niche-vehicle-dataset"
    )
    print(f"✅  Raw data at: {raw_path}")

    # -- Convert VOC XML → YOLO format -----------------------------------------
    img_dir  = _find_dir(raw_path, "auto")
    xml_dir  = _find_dir(raw_path, "Annotations")

    if not img_dir or not xml_dir:
        sys.exit("❌  Could not locate image/annotation directories in downloaded dataset.")

    train_img = os.path.join(DATA_DIR, "train", "images")
    train_lbl = os.path.join(DATA_DIR, "train", "labels")
    valid_img = os.path.join(DATA_DIR, "valid", "images")
    valid_lbl = os.path.join(DATA_DIR, "valid", "labels")
    for d in (train_img, train_lbl, valid_img, valid_lbl):
        os.makedirs(d, exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    random.seed(42)
    random.shuffle(xml_files)
    split = int(0.8 * len(xml_files))
    splits = {"train": xml_files[:split], "valid": xml_files[split:]}

    # auto/autorickshaw → class index 0 in this fallback dataset
    FALLBACK_CLASSES = {
        "auto": 0, "autorickshaw": 0, "auto rickshaw": 0,
        "e_rickshaw": 0, "e-rickshaw": 0, "rickshaw": 0,
        "car": 3, "vehicle": 3,
        "bus": 2, "truck": 8,
        "bike": 4, "motorcycle": 4, "two_wheeler": 4,
        "bicycle": 1, "cycle": 1,
        "person": 5, "pedestrian": 5,
        "tractor": 7,
    }

    converted = 0
    for subset, files in splits.items():
        for xml_file in files:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            fname = root.findtext("filename") or xml_file.replace(".xml", ".jpg")
            size  = root.find("size")
            if size is None:
                continue
            W, H = int(size.findtext("width", 640)), int(size.findtext("height", 480))

            lines = []
            for obj in root.iter("object"):
                name = (obj.findtext("name") or "").lower().strip()
                cls  = FALLBACK_CLASSES.get(name, None)
                if cls is None:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.findtext("xmin", 0))
                xmax = float(bb.findtext("xmax", W))
                ymin = float(bb.findtext("ymin", 0))
                ymax = float(bb.findtext("ymax", H))
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                continue

            # Copy image
            src_img = os.path.join(img_dir, fname)
            dst_img = os.path.join(DATA_DIR, subset, "images", fname)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            # Write label
            label_name = os.path.splitext(fname)[0] + ".txt"
            with open(os.path.join(DATA_DIR, subset, "labels", label_name), "w") as lf:
                lf.write("\n".join(lines))
            converted += 1

    print(f"✅  Converted {converted} annotated images to YOLO format.")
    return DATA_DIR


def _find_dir(root: str, target: str) -> str:
    """Find first directory whose name contains `target` (case-insensitive)."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if target.lower() in d.lower():
                full = os.path.join(dirpath, d)
                # Make sure it actually has files
                if any(os.scandir(full)):
                    return full
    return ""


# ── write YOLO data YAML ──────────────────────────────────────────────────────

def write_yaml(data_dir: str):
    """Write configs/indian_vehicles.yaml pointing at the downloaded data."""
    # Detect actual subfolder structure
    train_path = _find_split(data_dir, "train")
    valid_path = _find_split(data_dir, "valid")

    # YAML uses paths relative to itself OR absolute
    cfg = {
        "path"  : data_dir,
        "train" : os.path.relpath(train_path, data_dir) if train_path else "train/images",
        "val"   : os.path.relpath(valid_path, data_dir) if valid_path else "valid/images",
        "nc"    : len(CLASS_NAMES),
        "names" : CLASS_NAMES,
    }

    os.makedirs(CFG_DIR, exist_ok=True)
    yaml_path = os.path.join(CFG_DIR, "indian_vehicles.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"✅  Config written: {yaml_path}")
    return yaml_path


def _find_split(data_dir: str, split: str) -> str:
    """Locate train or valid images folder."""
    for candidate in [
        os.path.join(data_dir, split, "images"),
        os.path.join(data_dir, f"{split}_images"),
        os.path.join(data_dir, split),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return ""


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("🚦  Indian Vehicle Dataset Downloader")
    print("="*50)

    api_key  = _get_api_key()
    data_loc = download_roboflow(api_key)
    yaml_path = write_yaml(data_loc)

    # Count images for feedback
    total = sum(
        len([f for f in os.listdir(os.path.join(data_loc, s, "images"))
             if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        for s in ("train", "valid")
        if os.path.isdir(os.path.join(data_loc, s, "images"))
    )
    print(f"\n🎉  Done!  {total} images ready for training.")
    print(f"    Data  : {data_loc}")
    print(f"    Config: {yaml_path}")
    print(f"\n    Next: python src/train_yolo.py")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
