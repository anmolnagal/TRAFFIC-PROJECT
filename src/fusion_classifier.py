import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from preprocessing import preprocess_image
from frequency_features import extract_freq_features

class MockClassifier:
    def __init__(self, cls_id):
        self.cls_id = cls_id
        self.classes_ = np.array([cls_id])
    def predict(self, X):
        return np.full(X.shape[0], self.cls_id)
    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC, avgpool for 512 feat
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.pool(x).flatten(1)

extractor = FeatureExtractor()
extractor.eval()

def get_cnn_features(roi):
    """Extract CNN deep features from ROI."""
    processed, _ = preprocess_image(roi)
    processed = torch.from_numpy(processed).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        feats = extractor(processed).numpy()
    return feats.flatten()

def train_fusion(X_cnn, X_freq, y):
    """Train SVM on concatenated features."""
    # Use a single scaler fitted on the full fused feature vector
    X_fused = np.hstack([X_cnn, X_freq])
    scaler = StandardScaler()
    X_fused_scaled = scaler.fit_transform(X_fused)
    
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Warning: Only 1 class found in dataset ({unique_classes[0]}). Creating MockClassifier.")
        clf = MockClassifier(unique_classes[0])
    else:
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_fused_scaled, y)
        
    joblib.dump(clf, 'models/fusion_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Fusion classifier and scaler saved.")

def predict_fusion(bboxes, classes, confs, img):
    """Inference: extract feats for detected ROIs, classify."""
    scaler = joblib.load('models/scaler.pkl')
    clf = joblib.load('models/fusion_classifier.pkl')
    refined_classes = []
    for (x1,y1,x2,y2), cls, conf in zip(bboxes, classes, confs):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: continue
        cnn_feat = get_cnn_features(roi)
        freq_feat = extract_freq_features(roi)
        feats = scaler.transform(np.hstack([cnn_feat, freq_feat]).reshape(1,-1))
        pred_cls = int(clf.predict(feats)[0])
        class_names = {
            0: 'car', 1: 'two_wheeler', 2: 'autorickshaw', 3: 'e_autorickshaw',
            4: 'e_rickshaw', 5: 'bus', 6: 'electric_bus', 7: 'truck',
            8: 'tractor', 9: 'cycle', 10: 'pedestrian'
        }
        refined_classes.append(class_names.get(pred_cls, 'vehicle'))
    return bboxes, refined_classes, confs
