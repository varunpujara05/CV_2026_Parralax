"""
Step 12: Train YOLOv11m on Weather-Augmented Dataset
Trains a small-size YOLOv11 model on the combined clean + weather-augmented dataset
for improved detection robustness under adverse weather conditions.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix CUDA DLL path for Windows
try:
    import torch as _torch
    torch_lib = os.path.join(os.path.dirname(_torch.__file__), 'lib')
    if os.path.isdir(torch_lib):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
        if torch_lib not in os.environ.get('PATH', ''):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
except Exception:
    pass

from pathlib import Path


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = project_root / 'config' / 'visdrone_weather.yaml'
    
    print("=" * 60)
    print("STEP 12: TRAIN YOLOv11m ON WEATHER-AUGMENTED DATA")
    print("=" * 60)
    
    if not config_path.exists():
        print(f"ERROR: Weather config not found at {config_path}")
        print("Please run 11_prepare_weather_training.py first!")
        return
    
    # Check for GPU
    try:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory: {gpu_mem:.1f} GB")
        else:
            print("GPU: Not available (using CPU — training will be VERY slow)")
    except ImportError:
        device = 'cpu'
        print("PyTorch not found, using CPU")
    
    # Batch size — YOLOv11s fits batch=4 on 4GB VRAM
    batch_size = 4 if device != 'cpu' else 2
    
    print(f"\nConfig: {config_path}")
    print(f"Model: YOLOv11s (small)")
    print(f"Epochs: 10")
    print(f"Image size: 640")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Train YOLOv11m
    from ultralytics import YOLO
    
    model = YOLO('yolo11s.pt')  # YOLOv11 small
    
    print("\n--- Starting Training ---")
    results = model.train(
        data=str(config_path),
        epochs=10,
        imgsz=640,
        batch=batch_size,
        project=str(project_root / 'runs' / 'detect'),
        name='visdrone_weather_v11s',
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        patience=15,
        workers=0,
        device=device,
    )
    
    best_model = project_root / 'runs' / 'detect' / 'visdrone_weather_v11s' / 'weights' / 'best.pt'
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved to: {best_model}")
    print(f"\nNext step: Run 13_improved_pipeline.py for tracking + evaluation")


if __name__ == '__main__':
    main()
