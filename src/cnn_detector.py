from ultralytics import YOLO
import torch
from preprocessing import preprocess_image
import yaml

def load_config():
    # As above
    pass

config = load_config()
model_path = 'models/cnn_baseline.pt'

def train_cnn(data_yaml='data/dataset.yaml'):  # Create dataset.yaml for YOLO
    model = YOLO('yolov8n.pt')  # Nano for speed
    model.train(data=data_yaml, epochs=config['epochs'], imgsz=config['img_size'])
    model.save(model_path)

def detect(img_path):
    """Infer on single image, return bboxes, classes, confs."""
    model = YOLO(model_path)
    results = model(img_path)[0]
    bboxes = []
    classes = []
    confs = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        classes.append(config['classes'][cls])
        confs.append(float(conf))
    return bboxes, classes, confs
