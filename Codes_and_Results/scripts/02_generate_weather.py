"""
Step 2: Synthetic Weather Generation
Apply rain, fog, and dust augmentations to selected sequences.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weather_engine import (
    augment_sequence, generate_comparison_grid, apply_weather
)
from pathlib import Path
from tqdm import tqdm


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    annotations_dir = dataset_root / 'annotations'
    output_dir = project_root / 'outputs' / 'augmented'
    
    print("=" * 60)
    print("STEP 2: SYNTHETIC WEATHER GENERATION")
    print("=" * 60)
    
    # Select sequences for augmentation (5 representative sequences)
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    selected_seqs = all_seqs[:5]
    
    weather_types = ['rain', 'fog', 'dust']
    intensities = ['light', 'moderate', 'severe']
    
    print(f"\nSequences to augment: {len(selected_seqs)}")
    for s in selected_seqs:
        print(f"  - {s}")
    print(f"Weather types: {weather_types}")
    print(f"Intensities: {intensities}")
    
    # 1. Generate comparison grid from sample frame
    print("\n--- Generating Weather Comparison Grid ---")
    
    first_seq = sequences_dir / selected_seqs[0]
    first_frame_path = sorted(first_seq.glob('*.jpg'))[0]
    sample_img = cv2.imread(str(first_frame_path))
    
    grid_path = output_dir / 'weather_comparison_grid.jpg'
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    generate_comparison_grid(sample_img, str(grid_path))
    print(f"Comparison grid saved: {grid_path}")
    
    # 2. Generate individual weather samples
    print("\n--- Generating Individual Weather Samples ---")
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    for wt in weather_types:
        for intensity in intensities:
            augmented = apply_weather(sample_img, wt, intensity)
            sample_path = samples_dir / f'{wt}_{intensity}.jpg'
            cv2.imwrite(str(sample_path), augmented)
    print(f"Individual samples saved to: {samples_dir}")
    
    # 3. Augment selected sequences
    print("\n--- Augmenting Sequences ---")
    total_combos = len(selected_seqs) * len(weather_types) * len(intensities)
    combo_count = 0
    
    for seq_name in selected_seqs:
        seq_dir = sequences_dir / seq_name
        ann_path = annotations_dir / f'{seq_name}.txt'
        
        for wt in weather_types:
            for intensity in intensities:
                combo_count += 1
                print(f"\n[{combo_count}/{total_combos}] {seq_name} → {wt}_{intensity}")
                
                augment_sequence(
                    str(seq_dir),
                    str(ann_path),
                    str(output_dir),
                    wt,
                    intensity
                )
    
    print("\n" + "=" * 60)
    print("WEATHER AUGMENTATION COMPLETE!")
    print("=" * 60)
    print(f"\nAugmented data: {output_dir}")
    print(f"Total combinations: {total_combos}")


if __name__ == '__main__':
    main()
