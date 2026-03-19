"""
Step 4: Run Detection (Baseline + SAHI)
Run YOLOv8 inference on original and augmented sequences.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import detect_baseline, detect_sahi
from pathlib import Path


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try to find best model from training
    model_path = project_root / 'runs' / 'detect' / 'visdrone_train' / 'weights' / 'best.pt'
    if not model_path.exists():
        print("Trained model not found. Using pretrained YOLOv8s...")
        model_path = 'yolov8s.pt'
    
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    output_dir = project_root / 'outputs' / 'detections'
    augmented_dir = project_root / 'outputs' / 'augmented'
    
    print("=" * 60)
    print("STEP 4: RUN DETECTION")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    
    # Check device
    try:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    # Select sequences for evaluation
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    eval_seqs = all_seqs[:5]  # Match weather-augmented sequences
    
    conditions = [
        ('original', None, None),
        ('rain_light', 'rain', 'light'),
        ('rain_moderate', 'rain', 'moderate'),
        ('rain_severe', 'rain', 'severe'),
        ('fog_light', 'fog', 'light'),
        ('fog_moderate', 'fog', 'moderate'),
        ('fog_severe', 'fog', 'severe'),
        ('dust_light', 'dust', 'light'),
        ('dust_moderate', 'dust', 'moderate'),
        ('dust_severe', 'dust', 'severe'),
    ]
    
    for condition_name, weather, intensity in conditions:
        print(f"\n--- Detection: {condition_name} ---")
        
        for seq_name in eval_seqs:
            if weather and intensity:
                img_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
            else:
                img_dir = sequences_dir / seq_name
            
            if not img_dir.exists():
                print(f"  Skipping {seq_name} (not found)")
                continue
            
            # Baseline detection
            print(f"  Baseline: {seq_name}")
            det_out = output_dir / condition_name / 'baseline' / seq_name
            detect_baseline(
                model_path=str(model_path),
                image_dir=str(img_dir),
                output_dir=str(det_out),
                conf_thresh=0.25,
                device=device
            )
            
            # SAHI detection
            print(f"  SAHI: {seq_name}")
            det_out_sahi = output_dir / condition_name / 'sahi' / seq_name
            detect_sahi(
                model_path=str(model_path),
                image_dir=str(img_dir),
                output_dir=str(det_out_sahi),
                conf_thresh=0.25,
                slice_size=320,
                device=device
            )
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
