import os
import xml.etree.ElementTree as ET
import random
import shutil
import yaml

# Map classes to indices. Our model will be trained on these specific classes.
CLASS_MAPPING = {
    'car': 0,
    'two_wheeler': 1,
    'autorickshaw': 2,
    'e_autorickshaw': 3,
    'e_rickshaw': 4,
    'bus': 5,
    'electric_bus': 6,
    'truck': 7,
    'tractor': 8,
    'cycle': 9,
    'pedestrian': 10
}

def convert_to_yolo_format(size, box):
    # Normalized coordinates [0, 1]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def prepare_yolo_dataset(raw_dir, out_dir):
    img_dir = os.path.join(raw_dir, 'auto', 'auto')
    xml_dir = os.path.join(raw_dir, 'Annotations', 'Annotations')
    
    os.makedirs(os.path.join(out_dir, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train_labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val_images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val_labels"), exist_ok=True)
    
    # Get all XML files
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    random.seed(42)
    random.shuffle(xml_files)
    
    split_idx = int(0.8 * len(xml_files))
    train_files = xml_files[:split_idx]
    
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        
        # Image info
        img_filename = root.find('filename').text
        size_node = root.find('size')
        w = int(size_node.find('width').text)
        h = int(size_node.find('height').text)
        
        # Determine split
        is_train = xml_file in train_files
        subset = "train" if is_train else "val"
        
        # Write labels
        label_filename = xml_file.replace('.xml', '.txt')
        label_path = os.path.join(out_dir, f"{subset}_labels", label_filename)
        
        with open(label_path, 'w') as out_f:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text.lower()
                if cls_name not in CLASS_MAPPING:
                    # Generic auto assignment if named weird
                    if 'auto' in cls_name:
                        cls_name = 'autorickshaw'
                    else:
                        continue
                        
                cls_id = CLASS_MAPPING[cls_name]
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert_to_yolo_format((w, h), b)
                out_f.write(f"{cls_id} {' '.join(str(a) for a in bb)}\n")
        
        # Copy image
        src_img = os.path.join(img_dir, img_filename)
        dst_img = os.path.join(out_dir, f"{subset}_images", img_filename)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

def create_yaml_config():
    config = {
        'path': os.path.abspath('data'),
        'train': 'train_images',
        'val': 'val_images',
        'names': {v: k for k, v in CLASS_MAPPING.items()}
    }
    os.makedirs('configs', exist_ok=True)
    with open('configs/indian_vehicles.yaml', 'w') as f:
        yaml.dump(config, f)
    print("Created configs/indian_vehicles.yaml")

if __name__ == "__main__":
    raw_dir = "data/raw_dataset/autorickshaws"
    out_dir = "data"
    print("Converting VOC to YOLO format...")
    prepare_yolo_dataset(raw_dir, out_dir)
    print("Conversion complete.")
    create_yaml_config()
