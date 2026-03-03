import os
import cv2
import yaml
from pathlib import Path
import argparse

def load_config(config_path='configs/config.yaml'):
    """Load config."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def extract_frames(video_path, output_dir, every_n=30):  # Every 1 sec @30fps
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n == 0:
            cv2.imwrite(os.path.join(output_dir, f'frame_{saved_count:06d}.jpg'), frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f'Extracted {saved_count} frames to {output_dir}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='data/raw_videos')
    parser.add_argument('--output_dir', default='data/train_images')
    args = parser.parse_args()
    
    for vid in Path(args.video_dir).glob('*.mp4'):
        extract_frames(str(vid), args.output_dir)

if __name__ == '__main__':
    main()
