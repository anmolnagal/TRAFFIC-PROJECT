from ultralytics import YOLO
import yaml
import os
from preprocessing import preprocess_image, config

model_path = 'models/cnn_baseline.pt'

def load_config():
    """Load or create config."""
    config_path = 'configs/config.yaml'
    if not os.path.exists(config_path):
        os.makedirs('configs', exist_ok=True)
        config_data = {
            'img_size': 640,
            'classes': ['car', 'two_wheeler', 'auto_rickshaw', 'e_autorickshaw', 
                       'e_rickshaw', 'bus', 'electric_bus', 'truck', 'tractor', 'cycle', 'pedestrian']
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        return config_data
    with open(config_path) as f:
        return yaml.safe_load(f)

def train_cnn(data_yaml='data/dataset.yaml'):
    """Train custom YOLOv8 model."""
    model = YOLO('yolov8n.pt')  # Start with pre-trained nano model
    model.train(data=data_yaml, epochs=50, imgsz=config['img_size'])
    model.save(model_path)
    print(f"✅ Trained model saved to {model_path}")

def detect(img_path):
    """Detect vehicles - auto-fallback to pre-trained if no custom model."""
    global model_path
    
    # Try custom model first, fallback to pre-trained
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("✅ Using custom trained model")
    else:
        model = YOLO('yolov8n.pt')  # Pre-trained COCO model
        print("⚠️ Using pre-trained YOLOv8n (no custom model found)")
    
    results = model(img_path)[0]
    bboxes = []
    classes = []
    confs = []
    
    # Map COCO classes to our vehicle classes (basic mapping)
    coco_to_vehicle = {
        2: 'car', 3: 'car', 5: 'bus', 7: 'truck', 
        0: 'pedestrian'  # person
    }
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        
        # Map to vehicle class or use generic
        vehicle_class = coco_to_vehicle.get(cls_id, 'vehicle')
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        classes.append(vehicle_class)
        confs.append(conf)
    
    return bboxes, classes, confs

if __name__ == '__main__':
    # Test detection
    bboxes, classes, confs = detect('data/test_images/sample_traffic.jpg')
    print(f"Detected {len(bboxes)} objects: {classes}")
