"""
Step 1: Dataset Preparation
Parse VisDrone annotations and convert to YOLO detection format.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_utils import (
    visdrone_mot_to_yolo_detection,
    get_sequence_info,
    create_mot_ground_truth
)
from src.detection import create_yolo_config
from pathlib import Path


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    yolo_data_dir = project_root / 'data' / 'visdrone_yolo'
    gt_dir = project_root / 'outputs' / 'gt_mot'
    config_dir = project_root / 'config'
    
    # 1. Print dataset info
    print("=" * 60)
    print("STEP 1: DATASET PREPARATION")
    print("=" * 60)
    
    print("\n--- Dataset Information ---")
    info = get_sequence_info(str(dataset_root))
    
    total_frames = sum(s['num_frames'] for s in info)
    total_ids = sum(s['num_unique_ids'] for s in info)
    total_anns = sum(s['total_annotations'] for s in info)
    
    print(f"Sequences: {len(info)}")
    print(f"Total frames: {total_frames}")
    print(f"Total unique object IDs: {total_ids}")
    print(f"Total annotations: {total_anns}")
    
    print("\nSample sequences:")
    for seq in info[:5]:
        print(f"  {seq['name']}: {seq['num_frames']} frames, "
              f"{seq['num_unique_ids']} IDs, "
              f"{seq['total_annotations']} annotations")
    
    # 2. Convert to YOLO format
    print("\n--- Converting to YOLO Format ---")
    stats = visdrone_mot_to_yolo_detection(
        str(dataset_root),
        str(yolo_data_dir),
        val_ratio=0.2,
        sample_rate=5,  # Every 5th frame
        min_bbox_area=100
    )
    
    # 3. Create YOLO config
    print("\n--- Creating YOLO Config ---")
    config_path = create_yolo_config(str(config_dir), str(yolo_data_dir))
    print(f"Config: {config_path}")
    
    # 4. Create MOT ground truth files
    print("\n--- Creating MOT Ground Truth ---")
    create_mot_ground_truth(str(dataset_root), str(gt_dir))
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nYOLO dataset: {yolo_data_dir}")
    print(f"YOLO config: {config_path}")
    print(f"MOT ground truth: {gt_dir}")


if __name__ == '__main__':
    main()
