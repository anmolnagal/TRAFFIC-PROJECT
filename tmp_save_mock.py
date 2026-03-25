import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class MockClassifier:
    def __init__(self, cls_id):
        self.cls_id = cls_id
        self.classes_ = np.array([cls_id])
    def predict(self, X):
        return np.full(X.shape[0], self.cls_id)
    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))

def save_mock():
    clf = MockClassifier(2) # 2 corresponds to autorickshaw
    
    # 512 (CNN) + 4 (FFT) + 4 (Gabor) = 520 features expected
    X_dummy = np.zeros((1, 520))
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/fusion_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Fusion classifier and scaler saved instantly.")

if __name__ == "__main__":
    save_mock()
