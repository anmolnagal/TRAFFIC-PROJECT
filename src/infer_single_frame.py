import sys
import json
import cv2
from cnn_detector import detect
from fusion_classifier import predict_fusion

if __name__ == '__main__':
    img_path = sys.argv[1]
    out_json = sys.argv[2]
    
    img = cv2.imread(img_path)
    bboxes, classes, confs = detect(img_path)
    try:
        bboxes, classes, confs = predict_fusion(bboxes, classes, confs, img)
    except Exception as e:
        # Fusion model artifacts are optional; fall back to YOLO-only output.
        print(f"Fusion refinement unavailable (using YOLO classes only): {e}")
    
    results = {
        'bboxes': bboxes,
        'classes': classes,
        'confs': confs
    }
    with open(out_json, 'w') as f:
        json.dump(results, f)
