import os
import sys

# Windows DLL loading fix for PyTorch (c10.dll)
if os.name == 'nt':
    import torch
    dll_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(dll_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_path)

import glob
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

import torch
# Bypass PyTorch weights_only issue for ultralytics
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

import sys
sys.path.append('src')
from fusion_classifier import get_cnn_features
from frequency_features import extract_freq_features

# Mappings from COCO ID to our custom dataset IDs
COCO_TO_CUSTOM = {
    0: 10,   # person -> pedestrian
    2: 0,    # car -> car
    3: 1,    # motorcycle -> two_wheeler
    5: 5,    # bus -> bus
    7: 7     # truck -> truck
}

def train_multiclass():
    print("🚀 Starting Multi-Class SVM Training Pipeline...")
    X = []
    y = []
    
    # ---------------------------------------------------------
    # 1. Load Ground Truth Autorickshaws (Class 2)
    # ---------------------------------------------------------
    print("\n🔍 Extracting Ground-Truth Autos...")
    base_dir = os.getcwd()
    val_labels = glob.glob('data/val_labels/*.txt')
    # Limit ground truth to avoid overwhelming background classes
    import random
    random.shuffle(val_labels)
    val_labels = val_labels[:100] 
    print(f"Using {len(val_labels)} auto labels for balance.")
    for label_path in val_labels:
        img_path = label_path.replace('labels', 'images').replace('.txt', '.jpg')
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # format: cls x_center y_center width height
                cls_id = int(parts[0])
                if cls_id != 2: continue # Only autorickshaws
                
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                roi = img[y1:y2, x1:x2]
                if roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue
                    
                cnn_feat = get_cnn_features(roi)
                freq_feat = extract_freq_features(roi)
                X.append(np.hstack([cnn_feat, freq_feat]))
                y.append(2) # autorickshaw

    # ---------------------------------------------------------
    # 2. Load Pseudo-labeled Background Traffic 
    # ---------------------------------------------------------
    print(f"\n🚙 Extracting Background Traffic Crops via YOLOv8n...")
    print(f"\n🚙 Extracting Background Traffic Crops via YOLOv8n (conf=0.2)...")
    model = YOLO('yolov8n.pt')
    
    # Search all image folders for background samples
    search_patterns = [
        'data/test_images/*.jpg', 
        'data/val_images/*.jpg',
        'data/raw_dataset/**/*.jpg'
    ]
    test_images = []
    for p in search_patterns:
        test_images.extend(glob.glob(p, recursive=True))
    
    # Unique images for speed
    test_images = list(set(test_images))
    print(f"Checking {len(test_images)} images for background samples...")
    
    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None: continue
        
        results = model(img, conf=0.3)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().item())
            if cls_id in COCO_TO_CUSTOM:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                roi = img[y1:y2, x1:x2]
                if roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue
                    
                custom_id = COCO_TO_CUSTOM[cls_id]
                cnn_feat = get_cnn_features(roi)
                freq_feat = extract_freq_features(roi)
                X.append(np.hstack([cnn_feat, freq_feat]))
                y.append(custom_id)
                print(f" -> Grabbed {custom_id} from {os.path.basename(img_path)}")
                
    # ---------------------------------------------------------
    # 3. Train the Final SVM 
    # ---------------------------------------------------------
    print(f"\n🧠 Training SVM on {len(X)} diverse vehicle crops...")
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    clf.fit(X_scaled, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/fusion_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ Multi-Class Fusion Classifier & Scaler Saved!")
    print("Classes learned:", np.unique(y))

if __name__ == '__main__':
    train_multiclass()
