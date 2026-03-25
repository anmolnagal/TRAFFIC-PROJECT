import os
import cv2
import numpy as np
import yaml
from pathlib import Path

# Add src to path so we can import internal modules easily
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_classifier import get_cnn_features, train_fusion
from frequency_features import extract_freq_features

def unnormalize_bbox(w, h, nx, ny, nw, nh):
    """Convert YOLO [0, 1] normalized bounding box to image pixel coords."""
    x_center = float(nx) * w
    y_center = float(ny) * h
    width = float(nw) * w
    height = float(nh) * h
    
    x1 = int(round(x_center - width / 2.0))
    y1 = int(round(y_center - height / 2.0))
    x2 = int(round(x_center + width / 2.0))
    y2 = int(round(y_center + height / 2.0))
    
    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return x1, y1, x2, y2

def extract_all_training_features():
    train_images_dir = Path("data/train_images")
    train_labels_dir = Path("data/train_labels")
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print("Training directories not found. Please run download_datasets and prepare_yolo_dataset scripts first.")
        return
        
    X_cnn_list = []
    X_freq_list = []
    y_list = []
    
    images = list(train_images_dir.glob("*.jpg"))
    print(f"Found {len(images)} images for feature extraction. Starting...")
    
    for i, img_path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing image {i}/{len(images)}: {img_path.name}")
            
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            cls_id = int(parts[0])
            nx, ny, nw, nh = map(float, parts[1:])
            
            x1, y1, x2, y2 = unnormalize_bbox(w, h, nx, ny, nw, nh)
            
            roi = img[y1:y2, x1:x2]
            # Ensure ROI is valid
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                continue
                
            try:
                cnn_feat = get_cnn_features(roi)
                freq_feat = extract_freq_features(roi)
                
                X_cnn_list.append(cnn_feat)
                X_freq_list.append(freq_feat)
                y_list.append(cls_id)
            except Exception as e:
                print(f"Error processing ROI in {img_path.name}: {e}")
                
    if not X_cnn_list:
        print("No valid features extracted. Exiting.")
        return
        
    print(f"Extraction complete. Total ROIs: {len(y_list)}")
    print("Training SVM Fusion Classifier...")
    
    X_cnn = np.vstack(X_cnn_list)
    X_freq = np.vstack(X_freq_list)
    y = np.array(y_list)
    
    os.makedirs('models', exist_ok=True)
    train_fusion(X_cnn, X_freq, y)

if __name__ == "__main__":
    extract_all_training_features()
