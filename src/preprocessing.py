import cv2
import numpy as np
import yaml
import os


def load_config(config_path='configs/config.yaml'):
    """Load config - create if missing."""
    if not os.path.exists(config_path):
        # Auto-create basic config
        config = {
            'img_size': 640,
            'classes': ['car', 'two_wheeler', 'auto_rickshaw', 'e_autorickshaw', 
                       'e_rickshaw', 'bus', 'electric_bus', 'truck', 'tractor', 'cycle', 'pedestrian']
        }
        os.makedirs('configs', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Created {config_path}")
        return config
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

def preprocess_image(img, target_size=config['img_size']):
    """Resize and normalize."""
    h, w = img.shape[:2]
    scale = min(target_size/w, target_size/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas / 255.0, scale  # Normalize to [0,1]

def background_subtraction(img, bg_subtractor=None):
    """MOG2 background subtraction."""
    if bg_subtractor is None:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_subtractor.apply(img)
    return cv2.bitwise_and(img, img, mask=fg_mask)
