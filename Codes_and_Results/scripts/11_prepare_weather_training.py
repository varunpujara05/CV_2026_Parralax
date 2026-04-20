"""
Step 11: Prepare Weather-Augmented Training Dataset
Creates a combined clean + weather-augmented YOLO dataset for domain-adaptive training.
Weather augmentations (rain, fog, dust at light + moderate) are applied to training images.
Severe weather is reserved for testing only.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_utils import prepare_weather_augmented_yolo_dataset
from src.detection import create_yolo_config
from pathlib import Path


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    clean_yolo_dir = project_root / 'data' / 'visdrone_yolo'
    weather_yolo_dir = project_root / 'data' / 'visdrone_yolo_weather'
    config_dir = project_root / 'config'
    
    print("=" * 60)
    print("STEP 11: PREPARE WEATHER-AUGMENTED TRAINING DATASET")
    print("=" * 60)
    
    if not clean_yolo_dir.exists():
        print(f"ERROR: Clean YOLO dataset not found at {clean_yolo_dir}")
        print("Please run 01_prepare_dataset.py first!")
        return
    
    # Count existing clean images
    clean_train_count = len(list((clean_yolo_dir / 'images' / 'train').glob('*.jpg')))
    print(f"\nClean training images: {clean_train_count}")
    print(f"Weather types: rain, fog, dust")
    print(f"Intensities: light, moderate (severe reserved for testing)")
    print(f"Expected augmented dataset: ~{clean_train_count * 7}x images")
    print(f"  (1x clean + 6x augmented = 7x total)")
    
    # Prepare the combined dataset (all intensities including severe)
    stats = prepare_weather_augmented_yolo_dataset(
        clean_yolo_dir=str(clean_yolo_dir),
        output_dir=str(weather_yolo_dir),
        weather_types=['rain', 'fog', 'dust'],
        intensities=['light', 'moderate', 'severe']
    )
    
    # Create new YOLO config pointing to the weather-augmented dataset
    print("\n--- Creating Weather-Augmented YOLO Config ---")
    config_path = Path(config_dir) / 'visdrone_weather.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    weather_yolo_abs = weather_yolo_dir.resolve()
    config_content = f"""# VisDrone2019 MOT Dataset - Weather-Augmented YOLO Format
path: {str(weather_yolo_abs)}
train: images/train
val: images/val

# Number of classes
nc: 10

# Class names
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("WEATHER-AUGMENTED DATASET READY!")
    print("=" * 60)
    print(f"\nDataset: {weather_yolo_dir}")
    print(f"Config: {config_path}")
    print(f"\nNext step: Run 12_train_weather_detector.py to train YOLOv11m")


if __name__ == '__main__':
    main()
