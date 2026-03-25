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
        
    # Draw boxes
    for (x1, y1, x2, y2), cls, conf in zip(bboxes, classes, confs):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    out_img_path = img_path.replace(".jpg", "_result.jpg").replace(".png", "_result.png")
    cv2.imwrite(out_img_path, img)
    print(f"✅ Inference complete. JSON saved to {out_json}")
    print(f"📊 Visualization saved to {out_img_path}")
