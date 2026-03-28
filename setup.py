"""
setup.py  –  One-click setup for Indian Traffic Detection System
================================================================
This script does everything in order:
  1. Deletes old/broken project files
  2. Installs required Python packages
  3. Downloads the Indian Vehicles dataset from Roboflow
  4. Fine-tunes YOLOv8n on the dataset
  5. Verifies the trained model file exists

Run from the project root:
  python setup.py

After this completes, launch the GUI with:
  python demo/python_gui.py
"""

import os
import sys
import shutil
import subprocess

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT)

# ── Roboflow API Key (provided by user) ───────────────────────────────────────
# Get from env var if set, otherwise use the key below
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "zbIHn9nO9vykFxcpLSDg")

# ─────────────────────────────────────────────────────────────────────────────
def banner(text):
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}")

# ── Step 1: Delete dead/broken files ─────────────────────────────────────────
def cleanup():
    banner("Step 1 / 4 — Cleaning up old files")
    dead = [
        "src/train_multiclass_fusion.py",
        "src/train_fusion_classifier.py",
        "src/fusion_classifier.py",
        "src/cnn_detector.py",
        "src/frequency_features.py",
        "src/data_loader.py",
        "src/evaluator.py",
        "src/run_evaluater.py",
        "src/prepare_yolo_dataset.py",
        "src/preprocessing.py",
        "demo/download_sample_data.py",
        "demo/test_pipeline.py",
        "demo/matlab_gui.mlapp",
        "models/b0_cnn_baseline_broken.pt",
        "models/fusion_classifier.pkl",
        "models/scaler.pkl",
        "tmp_save_mock.py",
        "gui_history.json",
        "output.json",
    ]
    for f in dead:
        p = os.path.join(ROOT, f.replace("/", os.sep))
        if os.path.exists(p):
            os.remove(p)
            print(f"  🗑️   Deleted : {f}")
        else:
            print(f"  ⏭️   Skipped  : {f}  (not found)")

    pycache = os.path.join(ROOT, "src", "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
        print("  🗑️   Cleared : src/__pycache__")

    print("\n  ✅  Cleanup done.")


# ── Step 2: Install dependencies ─────────────────────────────────────────────
def install_deps():
    banner("Step 2 / 4 — Installing dependencies")
    reqs = os.path.join(ROOT, "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-r", reqs, "-q", "--upgrade"])
    print("  ✅  Dependencies installed.")


# ── Step 3: Download dataset ──────────────────────────────────────────────────
def download_dataset():
    banner("Step 3 / 4 — Downloading Indian Vehicles Dataset")

    os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY

    # Ensure roboflow package is available
    try:
        import roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])
        import roboflow

    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    # ── Try Dataset candidates in priority order ──────────────────────────────
    # Each entry: (workspace, project, version, description)
    candidates = [
        # Primary: Indian Vehicles (multi-class, ~5k images, YOLO format)
        ("indian-vehicles-obhqj", "indian-vehicles-qfkn2",    3, "Indian Vehicles v3"),
        ("sshikamaru",            "indian-vehicle-detection",  1, "Indian Vehicle Detection"),
        ("roboflow-universe-demos","indian-vehicles",          1, "RF Demo Indian Vehicles"),
        # Fallback: Autorickshaw only (confirmed working, ~8k images)
        ("dataclusterlabs",       "autorickshaw-detection",    1, "DataCluster Autorickshaw"),
    ]

    data_dir   = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    downloaded = False

    for ws, proj, ver, desc in candidates:
        try:
            print(f"\n  📥  Trying: {desc}  ({ws}/{proj} v{ver})")
            project = rf.workspace(ws).project(proj)
            dataset = project.version(ver).download("yolov8",
                                                     location=data_dir,
                                                     overwrite=True)
            print(f"  ✅  Downloaded to: {dataset.location}")
            downloaded = True
            _update_yaml(dataset.location)
            break
        except Exception as e:
            print(f"  ⚠️   Failed: {e}")
            continue

    if not downloaded:
        print("\n  ⚠️   All Roboflow sources failed → trying Kaggle autorickshaw fallback…")
        _kaggle_fallback(data_dir)

    print("\n  ✅  Dataset ready.")


def _update_yaml(dataset_location: str):
    """Read the downloaded data.yaml and copy/fix it to configs/."""
    import yaml, glob

    # Find the data.yaml inside the downloaded folder
    yaml_files = glob.glob(os.path.join(dataset_location, "*.yaml"))
    if not yaml_files:
        print("  ⚠️   No data.yaml found in downloaded dataset.")
        return

    with open(yaml_files[0]) as f:
        cfg = yaml.safe_load(f)

    # Fix the path to be absolute
    cfg["path"] = dataset_location

    # Ensure train/val keys exist
    if "train" not in cfg:
        cfg["train"] = "train/images"
    if "val" not in cfg and "valid" not in cfg:
        cfg["val"] = "valid/images"
    if "valid" in cfg and "val" not in cfg:
        cfg["val"] = cfg.pop("valid")

    cfg_path = os.path.join(ROOT, "configs", "indian_vehicles.yaml")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  ✅  Config updated: configs/indian_vehicles.yaml")

    # Print class info
    names = cfg.get("names", {})
    print(f"  📋  Classes ({cfg.get('nc', len(names))}): {list(names.values() if isinstance(names, dict) else names)}")


def _kaggle_fallback(data_dir: str):
    """Download autorickshaw dataset from Kaggle, convert VOC → YOLO."""
    import xml.etree.ElementTree as ET
    import random, yaml

    try:
        import kagglehub
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
        import kagglehub

    raw = kagglehub.dataset_download(
        "dataclusterlabs/autorickshaw-image-dataset-niche-vehicle-dataset")
    print(f"  ✅  Kaggle raw data at: {raw}")

    # Locate image/annotation dirs
    img_dir = xml_dir = ""
    for dp, dns, fns in os.walk(raw):
        if any(f.lower().endswith(".jpg") for f in fns) and not img_dir:
            img_dir = dp
        if any(f.lower().endswith(".xml") for f in fns) and not xml_dir:
            xml_dir = dp

    if not img_dir or not xml_dir:
        print("  ❌  Could not locate image/annotation dirs."); return

    CLASSES = {
        "auto":0,"autorickshaw":0,"auto rickshaw":0,"e_rickshaw":0,
        "rickshaw":0,"car":3,"bus":2,"truck":8,"motorcycle":4,
        "bike":4,"two_wheeler":4,"bicycle":1,"cycle":1,
        "person":5,"pedestrian":5,"tractor":7,
    }
    CLASS_NAMES = {0:"auto",1:"bicycle",2:"bus",3:"car",4:"motorcycle",
                   5:"pedestrian",6:"tempo",7:"tractor",8:"truck",9:"van"}

    for split in ("train","valid"):
        os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)

    xmls = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    random.seed(42); random.shuffle(xmls)
    split_idx = int(0.8 * len(xmls))
    assigned = {f: ("train" if i < split_idx else "valid") for i, f in enumerate(xmls)}

    ok = 0
    for xf, subset in assigned.items():
        tree = ET.parse(os.path.join(xml_dir, xf)); root = tree.getroot()
        fname = root.findtext("filename") or xf.replace(".xml",".jpg")
        sz    = root.find("size")
        W = int(sz.findtext("width",640)); H = int(sz.findtext("height",480))
        lines = []
        for obj in root.iter("object"):
            nm  = (obj.findtext("name") or "").lower().strip()
            cls = CLASSES.get(nm)
            if cls is None: continue
            bb  = obj.find("bndbox")
            if bb is None: continue
            xmin=float(bb.findtext("xmin",0)); xmax=float(bb.findtext("xmax",W))
            ymin=float(bb.findtext("ymin",0)); ymax=float(bb.findtext("ymax",H))
            cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
            bw=(xmax-xmin)/W;     bh=(ymax-ymin)/H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines: continue
        src=os.path.join(img_dir,fname); dst=os.path.join(data_dir,subset,"images",fname)
        if os.path.exists(src): shutil.copy2(src,dst)
        lbl=os.path.splitext(fname)[0]+".txt"
        with open(os.path.join(data_dir,subset,"labels",lbl),"w") as lf:
            lf.write("\n".join(lines))
        ok += 1

    print(f"  ✅  Converted {ok} images to YOLO format.")

    cfg = {"path": data_dir, "train":"train/images", "val":"valid/images",
           "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    cfg_path = os.path.join(ROOT, "configs", "indian_vehicles.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  ✅  Config written: {cfg_path}")


# ── Step 4: Train YOLOv8 ─────────────────────────────────────────────────────
def train_model():
    banner("Step 4 / 4 — Fine-tuning YOLOv8n on Indian Vehicles")
    train_script = os.path.join(ROOT, "src", "train_yolo.py")
    result = subprocess.run([sys.executable, train_script],
                            cwd=ROOT, check=False)
    if result.returncode == 0:
        model_path = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / 1_048_576
            print(f"\n  🎉  Training complete!")
            print(f"  📦  Model saved: models/indian_vehicles_yolo.pt  ({size_mb:.1f} MB)")
        else:
            print("  ⚠️   Model file not found — check runs/train/indian_vehicles/weights/")
    else:
        print("  ❌  Training failed. Check the output above for errors.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "🚦 " * 20)
    print("  INDIAN TRAFFIC DETECTION — Setup & Training")
    print("🚦 " * 20)

    try:
        cleanup()
        install_deps()
        download_dataset()
        train_model()

        print("\n" + "=" * 60)
        print("  🎉  ALL DONE!")
        print("  Launch the GUI:")
        print("      python demo/python_gui.py")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️   Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌  Setup failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
