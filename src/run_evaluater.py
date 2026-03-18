import os
import glob
import numpy as np
from preprocessing import load_config
from cnn_detector import detect
from fusion_classifier import predict_fusion
from evaluator import evaluate_detection, plot_confusion_matrix
import cv2

def _yolo_txt_path_for_image(img_path: str, labels_dir: str | None = None) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if labels_dir:
        return os.path.join(labels_dir, f"{stem}.txt")
    return os.path.join(os.path.dirname(img_path), f"{stem}.txt")

def _load_yolo_ground_truth(img_path: str, classes: list[str], labels_dir: str | None = None):
    """Load YOLO-format labels for one image and convert to pixel xyxy boxes.

    YOLO label format per line: class_id x_center y_center width height (all normalized 0..1)
    """
    img = cv2.imread(img_path)
    if img is None:
        return [], []
    h, w = img.shape[:2]

    label_path = _yolo_txt_path_for_image(img_path, labels_dir=labels_dir)
    if not os.path.exists(label_path):
        return [], []

    true_bboxes: list[list[int]] = []
    true_classes: list[str] = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_name = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
            true_bboxes.append([x1, y1, x2, y2])
            true_classes.append(cls_name)

    return true_bboxes, true_classes

def main():
    config = load_config()
    val_dir = config['val_path']
    classes = config['classes']
    val_labels_dir = config.get('val_labels_path')  # optional
    
    all_pred_bboxes = []
    all_pred_classes = []
    all_true_bboxes = []
    all_true_classes = []
    
    # NOTE: You will need to load your ground truth (true labels) from your dataset annotations.
    # For this example, we assume you have a function `get_ground_truth(img_path)`
    # that returns true_bboxes, true_classes for a given image.
    
    image_paths = glob.glob(os.path.join(val_dir, '*.jpg'))
    
    if not image_paths:
        print(f"❌ No validation images found in {val_dir}. Please add data.")
        return

    print(f"Starting evaluation on {len(image_paths)} images...")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 1. Get Predictions
        bboxes, pred_classes, confs = detect(img_path)
        try:
            pred_bboxes, final_classes, _ = predict_fusion(bboxes, pred_classes, confs, img)
        except Exception:
            pred_bboxes, final_classes = bboxes, pred_classes
        
        # 2. Get Ground Truth from YOLO txt labels
        true_bboxes, true_classes = _load_yolo_ground_truth(
            img_path,
            classes=classes,
            labels_dir=val_labels_dir,
        )
        
        # Store for evaluation
        all_pred_bboxes.extend(pred_bboxes)
        all_pred_classes.extend(final_classes)
        all_true_bboxes.extend(true_bboxes)
        all_true_classes.extend(true_classes)

    # 3. Calculate Metrics using your evaluator.py
    metrics = evaluate_detection(all_pred_bboxes, all_pred_classes, all_true_bboxes, all_true_classes, iou_th=0.5)
    
    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # 4. Generate Confusion Matrix
    if all_true_classes:
        plot_confusion_matrix(all_true_classes, all_pred_classes, classes)
        
if __name__ == '__main__':
    main()