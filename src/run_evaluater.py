import os
import glob
import numpy as np
from preprocessing import load_config
from cnn_detector import detect
from fusion_classifier import predict_fusion
from evaluator import evaluate_detection, plot_confusion_matrix
import cv2

def main():
    config = load_config()
    val_dir = config['val_path']
    classes = config['classes']
    
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
        
        # 1. Get Predictions
        bboxes, pred_classes, confs = detect(img_path)
        pred_bboxes, final_classes, _ = predict_fusion(bboxes, pred_classes, confs, img)
        
        # 2. Get Ground Truth (You must implement this based on your annotation format, e.g., YOLO txt files)
        # true_bboxes, true_classes = get_ground_truth(img_path) 
        
        # Placeholder for demonstration:
        true_bboxes, true_classes = [], [] 
        
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