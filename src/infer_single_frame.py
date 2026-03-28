"""
infer_single_frame.py
=====================
Run YOLO detection on a single image and write results to JSON.

Usage:
  python src/infer_single_frame.py <image_path> <output_json>

Falls back to pre-trained YOLOv8n (COCO) if the fine-tuned model doesn't
exist yet -- so the GUI works even before training is done.
"""

import sys
import json
import os
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# ── Model paths ───────────────────────────────────────────────────────────────
CUSTOM_MODEL  = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")
FALLBACK_MODEL = os.path.join(ROOT, "yolov8n.pt")

# COCO class → Indian vehicle name (for fallback mode with yolov8n.pt)
COCO_TO_INDIAN = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "two_wheeler",
    5: "bus",
    7: "truck",
}


def load_model():
    from ultralytics import YOLO
    if os.path.exists(CUSTOM_MODEL):
        print(f"Using fine-tuned model: {CUSTOM_MODEL}")
        return YOLO(CUSTOM_MODEL), True
    print(f"Fine-tuned model not found. Using fallback: {FALLBACK_MODEL}")
    return YOLO(FALLBACK_MODEL), False


def run_inference(img_path: str, out_json: str):
    model, is_custom = load_model()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    conf_thresh = 0.35 if is_custom else 0.40
    results = model(img_path, conf=conf_thresh, verbose=False)[0]

    bboxes, classes, confs = [], [], []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if is_custom:
            # Use model's own class names (Indian vehicles)
            class_name = results.names.get(cls_id, f"class_{cls_id}")
        else:
            # Filter to vehicle-related COCO classes only
            if cls_id not in COCO_TO_INDIAN:
                continue
            class_name = COCO_TO_INDIAN[cls_id]

        bboxes.append([x1, y1, x2, y2])
        classes.append(class_name)
        confs.append(round(conf, 4))

    output = {"bboxes": bboxes, "classes": classes, "confs": confs}
    with open(out_json, "w") as f:
        json.dump(output, f)

    print(f"✅  {len(bboxes)} detection(s) → {out_json}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/infer_single_frame.py <image_path> <output_json>")
        sys.exit(1)
    run_inference(sys.argv[1], sys.argv[2])
