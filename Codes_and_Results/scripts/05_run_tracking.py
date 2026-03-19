"""
Step 5: Run Multi-Object Tracking
Run ByteTrack and DeepSORT on detection results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking import run_tracking_on_sequence, save_tracks_mot_format
from pathlib import Path
from tqdm import tqdm


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Find model
    model_path = project_root / 'runs' / 'detect' / 'visdrone_train' / 'weights' / 'best.pt'
    if not model_path.exists():
        model_path = 'yolov8s.pt'
    
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    output_dir = project_root / 'outputs' / 'tracks'
    
    print("=" * 60)
    print("STEP 5: MULTI-OBJECT TRACKING")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    
    # Check device
    try:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    # Select sequences
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    eval_seqs = all_seqs[:5]
    
    tracker_types = ['bytetrack']  # Add 'deepsort' if boxmot is available
    sahi_modes = [False, True]
    
    conditions = [
        ('original', None, None),
        ('rain_severe', 'rain', 'severe'),
        ('fog_severe', 'fog', 'severe'),
        ('dust_severe', 'dust', 'severe'),
    ]
    
    for tracker_type in tracker_types:
        for use_sahi in sahi_modes:
            sahi_tag = 'sahi' if use_sahi else 'baseline'
            
            for condition_name, weather, intensity in conditions:
                experiment_name = f'{tracker_type}_{sahi_tag}_{condition_name}'
                print(f"\n--- {experiment_name} ---")
                
                for seq_name in eval_seqs:
                    if weather and intensity:
                        seq_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
                    else:
                        seq_dir = sequences_dir / seq_name
                    
                    if not seq_dir.exists():
                        print(f"  Skipping {seq_name}")
                        continue
                    
                    print(f"  Processing: {seq_name}")
                    
                    tracks = run_tracking_on_sequence(
                        model_path=str(model_path),
                        seq_dir=str(seq_dir),
                        tracker_type=tracker_type,
                        conf_thresh=0.25,
                        imgsz=640,
                        device=device,
                        use_sahi=use_sahi,
                        sahi_slice_size=320
                    )
                    
                    # Save tracks
                    track_file = output_dir / experiment_name / seq_name / f'{seq_name}.txt'
                    save_tracks_mot_format(tracks, str(track_file))
                    
                    print(f"    Total tracked objects: {sum(len(t) for t in tracks.values())}")
    
    print("\n" + "=" * 60)
    print("TRACKING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
