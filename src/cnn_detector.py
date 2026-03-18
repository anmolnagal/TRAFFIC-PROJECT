import os
from typing import Union

import numpy as np
import torch
from torch.nn.modules.container import ModuleList, Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.pooling import AdaptiveAvgPool2d

from ultralytics import YOLO

from preprocessing import config

model_path = 'models/cnn_baseline.pt'

def _ensure_ultralytics_torch_safe_load():
    """Allow-list Ultralytics/Torch classes for PyTorch 'weights_only' loading.

    PyTorch 2.6+ defaults torch.load(weights_only=True). Some checkpoints (including
    Ultralytics YOLO weights) require allow-listing certain classes for safe loading.
    """
    add_safe = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add_safe is None:
        return

    safe: list[object] = [
        Sequential,
        ModuleList,
        Conv2d,
        BatchNorm2d,
        SiLU,
        MaxPool2d,
        AdaptiveAvgPool2d,
        Upsample,
    ]

    try:
        from ultralytics.nn.tasks import DetectionModel  # type: ignore
        safe.append((DetectionModel, "ultralytics.nn.tasks.DetectionModel"))
    except Exception:
        pass

    # Ultralytics exposes most YOLOv8 layer blocks from `ultralytics.nn.modules`.
    # Allow-list all of them under their canonical module path to keep loading stable.
    try:
        import ultralytics.nn.modules as ulm  # type: ignore

        for name, obj in vars(ulm).items():
            if isinstance(obj, type):
                safe.append((obj, f"ultralytics.nn.modules.{name}"))
    except Exception:
        pass

    try:
        add_safe(safe)
    except Exception:
        return

def train_cnn(data_yaml: str = 'configs/dataset.yaml', epochs: int = 50):
    """Train custom YOLOv8 model."""
    _ensure_ultralytics_torch_safe_load()
    model = YOLO('yolov8n.pt')  # Start with pre-trained nano model
    model.train(data=data_yaml, epochs=epochs, imgsz=config['img_size'])
    model.save(model_path)
    print(f"Trained model saved to {model_path}")

def detect(img: Union[str, np.ndarray]):
    """Detect vehicles in an image path OR BGR image array.

    Auto-fallback to pre-trained YOLOv8n if no custom model found.
    """
    global model_path
    _ensure_ultralytics_torch_safe_load()
    
    # Try custom model first, fallback to pre-trained
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("Using custom trained model")
    else:
        model = YOLO('yolov8n.pt')  # Pre-trained COCO model
        print("Using pre-trained YOLOv8n (no custom model found)")
    
    if isinstance(img, str):
        results = model(img)[0]
    else:
        if not isinstance(img, np.ndarray):
            raise TypeError("detect() expects a file path (str) or an image (numpy.ndarray).")
        results = model(img)[0]
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
