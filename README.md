# Hybrid Feature-Based Vehicle Detection and Classification for Intelligent Traffic Analysis

![Status](https://img.shields.io/badge/Project--Status-Approved-brightgreen)
![Tech Stack](https://img.shields.io/badge/Technology--Stack-Python%20%7C%20MATLAB-blue)
![Type](https://img.shields.io/badge/Problem--Type-Deeptech%20%26%20System%20Based-orange)


## 👥 The Team

| Name | Role |
| :--- | :--- |
| **Anmol Nagal** | Project Leader |
| **Vivek Rana** | Team Member |
| **Riya Vashistha** | Team Member |
| **Shubham** | Team Member |
| **Adtiya** | Team Member |

---

## 📌 Project Overview
Accurate detection and classification of vehicles in real-world traffic scenes remain challenging due to varying illumination, occlusions, motion blur, and complex backgrounds. 

This project proposes a **hybrid computer vision framework** that combines:
* **CNN-based Spatial Features:** To capture structural and hierarchical visual information.
* **Handcrafted Frequency-Domain Descriptors:** To enhance robustness and generalization across diverse environments where spatial features alone might fail.



---

## 🎯 Objectives
1.  **System Fundamentals:** Understand the core principles of vision-based traffic analysis.
2.  **Preprocessing:** Apply image and video enhancement techniques specifically for traffic scenes.
3.  **Feature Extraction:** Implement CNN architectures to extract high-level spatial features.
4.  **Hybrid Integration:** Fuse deep features with handcrafted frequency-domain features.
5.  **Performance Evaluation:** Benchmark the system using standard detection and classification metrics.

---

## 🛠️ Technology Stack
* **Languages:** Python, MATLAB
* **Key Libraries:** OpenCV, PyTorch/TensorFlow, NumPy, Scikit-Image
* **Tools:** Git, MATLAB Image Processing Toolbox

---

## 📈 Proposed Solution & Workflow
1.  **Literature Review:** Researching traffic vision and hybrid feature fusion methodologies.
2.  **Data Preparation:** Preprocessing traffic image/video datasets for training and testing.
3.  **Model Implementation:** Building the hybrid vehicle detection and classification model.
4.  **Comparative Analysis:** Performance evaluation against traditional deep learning models.
5.  **Documentation:** Final project report and system demonstration.

---

## ▶️ Run it (Windows / PowerShell)

### Install

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Quick demo (download sample + test pipeline)

```bash
python demo\download_sample_data.py
python demo\test_pipeline.py
```

### Launch the Python GUI (Gradio)

```bash
python demo\python_gui.py
```

### Run single-image inference (writes JSON)

```bash
python src\infer_single_frame.py data\test_images\sample_traffic.jpg output.json
```

### Train YOLOv8 on your dataset (optional)

1. Put your YOLO dataset under:
   - `data/train_images`, `data/val_images`
   - labels either next to images (`.txt` beside each `.jpg`) **or** in `data/train_labels`, `data/val_labels`
2. Update `configs/dataset.yaml` if needed.

Then run:

```bash
python -c "import sys; sys.path.insert(0,'src'); from cnn_detector import train_cnn; train_cnn('configs/dataset.yaml', epochs=50)"
```

### Evaluate (requires YOLO `.txt` labels for validation images)

If your validation labels live in a separate folder, add this to `configs/config.yaml`:

```yaml
val_labels_path: data/val_labels
```

Then run:

```bash
python src\run_evaluater.py
```

## 🏁 Expected Outcomes
The integrated feature representation aims to significantly improve vehicle identification performance in adverse conditions such as:
* High-speed motion blur.
* Low-light or night-time traffic.
* Partially occluded vehicles in heavy congestion.

---
© 2026 | Developed as part of the Intelligent Traffic Analysis Research Project.
