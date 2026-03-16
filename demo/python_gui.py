import sys
import os
import cv2
import gradio as gr
import numpy as np

# ── Anchor all paths to the project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
os.chdir(PROJECT_ROOT)  

# Import your existing pipeline
from cnn_detector import detect
from fusion_classifier import predict_fusion

def process_traffic_image(image):
    """Processes the image through YOLO and the SVM fusion classifier."""
    if image is None:
        return None
        
    # Gradio provides an RGB image, but OpenCV/YOLO usually expect BGR.
    # We will save a temporary frame just to be fully compatible with your detect() function.
    temp_path = "temp_gui_frame.jpg"
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, bgr_img)
    
    try:
        # 1. Detect using YOLO
        bboxes, refined_classes, confs = detect(temp_path)
        
        # 2. Refine classification using your hybrid feature pipeline
        # bboxes, refined_classes, confs = predict_fusion(bboxes, classes, confs, bgr_img)
        
        # 3. Draw bounding boxes and labels on the image
        annotated_img = bgr_img.copy()
        for (x1, y1, x2, y2), cls, conf in zip(bboxes, refined_classes, confs):
            # Draw green rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label text
            label = f"{cls} ({conf:.2f})"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        # Convert back to RGB so Gradio displays colors correctly
        return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return image # Return original if it fails

# Create the Gradio Interface
interface = gr.Interface(
    fn=process_traffic_image,
    inputs=gr.Image(type="numpy", label="Upload Traffic Image"),
    outputs=gr.Image(type="numpy", label="Detected Vehicles"),
    title="Intelligent Traffic Analysis",
    description="Upload an image to detect and classify vehicles using the Hybrid CNN + Frequency pipeline.",
)

if __name__ == '__main__':
    print("🚀 Launching Python GUI...")
    interface.launch(inbrowser=True)