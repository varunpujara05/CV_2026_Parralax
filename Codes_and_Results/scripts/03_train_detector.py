"""
Step 3: Train YOLOv8 Detector
Fine-tune YOLOv8 on VisDrone dataset for small object detection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import train_yolo
from pathlib import Path


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = project_root / 'config' / 'visdrone.yaml'
    
    print("=" * 60)
    print("STEP 3: TRAIN YOLOv8 DETECTOR")
    print("=" * 60)
    
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        print("Please run 01_prepare_dataset.py first!")
        return
    
    # Check for GPU
    try:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: Not available (using CPU — training will be slow)")
    except ImportError:
        device = 'cpu'
        print("PyTorch not found, using CPU")
    
    print(f"\nConfig: {config_path}")
    print("Model: YOLOv8s (small)")
    print("Epochs: 20")
    print("Image size: 640")
    print(f"Batch size: {8 if device != 'cpu' else 4}")
    
    print("\n--- Starting Training ---")
    results = train_yolo(
        data_yaml=str(config_path),
        model_name='yolov8s.pt',
        epochs=20,
        imgsz=640,
        batch=8 if device != 'cpu' else 4,
        project=str(project_root / 'runs' / 'detect'),
        name='visdrone_train',
        device=device
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model: {project_root}/runs/detect/visdrone_train/weights/best.pt")


if __name__ == '__main__':
    main()
