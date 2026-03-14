import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
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
    scaler = StandardScaler()
    X_cnn = scaler.fit_transform(X_cnn)
    X_freq = scaler.transform(X_freq)  # Same scaler
    X_fused = np.hstack([X_cnn, X_freq])
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_fused, y)
    joblib.dump(clf, 'models/fusion_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

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
        pred_cls = clf.predict(feats)[0]
        refined_classes.append(pred_cls)
    return bboxes, refined_classes, confs
