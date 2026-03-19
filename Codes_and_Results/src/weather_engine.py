"""
Synthetic Weather Engine for UAV Video Augmentation
Simulates rain, fog, and dust storms with parameterized intensity levels.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import random


# ============================================================
#  INTENSITY CONFIGURATIONS
# ============================================================

RAIN_CONFIG = {
    'light': {
        'num_drops': 300, 'drop_length': 10, 'drop_thickness': 1,
        'brightness_factor': 0.90, 'blur_kernel': 3, 'drop_color': (200, 200, 200),
        'angle_range': (-10, 10)
    },
    'moderate': {
        'num_drops': 800, 'drop_length': 18, 'drop_thickness': 1,
        'brightness_factor': 0.78, 'blur_kernel': 5, 'drop_color': (180, 180, 180),
        'angle_range': (-15, 15)
    },
    'severe': {
        'num_drops': 1800, 'drop_length': 28, 'drop_thickness': 2,
        'brightness_factor': 0.60, 'blur_kernel': 7, 'drop_color': (160, 160, 160),
        'angle_range': (-20, 20)
    }
}

FOG_CONFIG = {
    'light': {
        'fog_intensity': 0.25, 'contrast_factor': 0.85,
        'blur_kernel': 5, 'fog_color': (220, 220, 220)
    },
    'moderate': {
        'fog_intensity': 0.45, 'contrast_factor': 0.65,
        'blur_kernel': 9, 'fog_color': (210, 210, 210)
    },
    'severe': {
        'fog_intensity': 0.70, 'contrast_factor': 0.40,
        'blur_kernel': 15, 'fog_color': (200, 200, 200)
    }
}

DUST_CONFIG = {
    'light': {
        'tint_color': (60, 120, 180), 'tint_strength': 0.15,
        'noise_sigma': 15, 'opacity': 0.10, 'blur_kernel': 3
    },
    'moderate': {
        'tint_color': (50, 100, 170), 'tint_strength': 0.30,
        'noise_sigma': 30, 'opacity': 0.25, 'blur_kernel': 5
    },
    'severe': {
        'tint_color': (40, 80, 150), 'tint_strength': 0.50,
        'noise_sigma': 50, 'opacity': 0.45, 'blur_kernel': 9
    }
}


# ============================================================
#  WEATHER SIMULATION FUNCTIONS
# ============================================================

def apply_rain(image, intensity='moderate'):
    """
    Apply rain effect to an image.
    
    Creates diagonal rain streaks, applies motion blur, and reduces brightness.
    """
    config = RAIN_CONFIG[intensity]
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Darken the image (overcast sky)
    result *= config['brightness_factor']
    
    # Create rain streak overlay
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(config['num_drops']):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        angle = random.randint(*config['angle_range'])
        length = config['drop_length'] + random.randint(-3, 3)
        
        x2 = x1 + int(length * np.sin(np.radians(angle)))
        y2 = y1 + length
        
        cv2.line(rain_layer, (x1, y1), (x2, y2), 255,
                 thickness=config['drop_thickness'],
                 lineType=cv2.LINE_AA)
    
    # Apply motion blur to rain streaks
    k = config['blur_kernel']
    kernel = np.zeros((k, k))
    kernel[:, k // 2] = 1.0 / k
    rain_layer = cv2.filter2D(rain_layer, -1, kernel)
    
    # Blend rain streaks onto image
    rain_mask = rain_layer.astype(np.float32) / 255.0
    rain_color = np.array(config['drop_color'], dtype=np.float32)
    
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - rain_mask * 0.6) + rain_color[c] * rain_mask * 0.6
    
    # Apply slight overall blur for wet atmosphere
    if config['blur_kernel'] > 3:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_fog(image, intensity='moderate'):
    """
    Apply fog effect to an image.
    
    Creates a haze layer with depth-based fog density and reduces contrast.
    """
    config = FOG_CONFIG[intensity]
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Reduce contrast
    mean = np.mean(result)
    result = mean + config['contrast_factor'] * (result - mean)
    
    # Create depth-based fog map (denser at bottom for UAV perspective)
    fog_map = np.ones((h, w), dtype=np.float32) * config['fog_intensity']
    
    # Add gradient: slightly less fog at top (closer objects in UAV view)
    gradient = np.linspace(0.7, 1.0, h).reshape(-1, 1)
    fog_map *= gradient
    
    # Add Perlin-like noise for natural look
    noise = np.random.normal(0, 0.05, (h // 4, w // 4)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    fog_map = np.clip(fog_map + noise * 0.1, 0, 1)
    
    # Apply Gaussian blur for smooth fog
    k = config['blur_kernel']
    fog_map = cv2.GaussianBlur(fog_map, (k, k), 0)
    
    # Blend fog color
    fog_color = np.array(config['fog_color'], dtype=np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - fog_map) + fog_color[c] * fog_map
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_dust(image, intensity='moderate'):
    """
    Apply dust storm effect to an image.
    
    Adds warm color tint, Gaussian noise, and opacity overlay.
    """
    config = DUST_CONFIG[intensity]
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Apply warm/orange color tint (BGR format)
    tint = np.full_like(result, config['tint_color'], dtype=np.float32)
    result = result * (1 - config['tint_strength']) + tint * config['tint_strength']
    
    # Add Gaussian noise
    noise = np.random.normal(0, config['noise_sigma'], result.shape).astype(np.float32)
    result += noise
    
    # Apply opacity overlay (dust particles)
    dust_layer = np.random.normal(128, 30, (h // 2, w // 2)).astype(np.float32)
    dust_layer = cv2.resize(dust_layer, (w, h), interpolation=cv2.INTER_LINEAR)
    k = config['blur_kernel']
    dust_layer = cv2.GaussianBlur(dust_layer, (k, k), 0)
    dust_layer = dust_layer / 255.0 * config['opacity']
    
    dust_color = np.array([130, 160, 200], dtype=np.float32)  # Sandy brown in BGR
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - dust_layer) + dust_color[c] * dust_layer
    
    # Slight blur for atmospheric haze
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
#  UNIFIED API
# ============================================================

WEATHER_FUNCTIONS = {
    'rain': apply_rain,
    'fog': apply_fog,
    'dust': apply_dust
}


def apply_weather(image, weather_type, intensity='moderate'):
    """
    Apply weather augmentation to a single image.
    
    Args:
        image: numpy array (BGR)
        weather_type: 'rain', 'fog', or 'dust'
        intensity: 'light', 'moderate', or 'severe'
        
    Returns:
        Augmented image (numpy array, BGR)
    """
    if weather_type not in WEATHER_FUNCTIONS:
        raise ValueError(f"Unknown weather type: {weather_type}. Use: {list(WEATHER_FUNCTIONS.keys())}")
    if intensity not in ['light', 'moderate', 'severe']:
        raise ValueError(f"Unknown intensity: {intensity}. Use: light, moderate, severe")
    
    return WEATHER_FUNCTIONS[weather_type](image, intensity)


def augment_sequence(seq_dir, ann_path, output_base, weather_type, intensity,
                     copy_annotations=True):
    """
    Apply weather augmentation to an entire video sequence.
    
    Args:
        seq_dir: Path to sequence frame directory
        ann_path: Path to annotation file
        output_base: Base output directory
        weather_type: 'rain', 'fog', or 'dust'
        intensity: 'light', 'moderate', or 'severe'
        copy_annotations: Whether to copy annotations (bboxes don't change)
    """
    seq_dir = Path(seq_dir)
    ann_path = Path(ann_path)
    output_base = Path(output_base)
    
    seq_name = seq_dir.name
    out_seq_dir = output_base / f'{weather_type}_{intensity}' / 'sequences' / seq_name
    out_ann_dir = output_base / f'{weather_type}_{intensity}' / 'annotations'
    
    out_seq_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frames
    frame_files = sorted(seq_dir.glob('*.jpg'))
    
    for frame_path in tqdm(frame_files, desc=f'{weather_type}_{intensity}/{seq_name}',
                           leave=False):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Apply weather augmentation
        augmented = apply_weather(img, weather_type, intensity)
        
        # Save augmented frame
        cv2.imwrite(str(out_seq_dir / frame_path.name), augmented)
    
    # Copy annotations (bounding boxes don't change with weather)
    if copy_annotations and ann_path.exists():
        shutil.copy2(ann_path, out_ann_dir / ann_path.name)


def generate_comparison_grid(image, save_path=None):
    """
    Generate a 4x3 comparison grid showing all weather types and intensities.
    
    Args:
        image: Original image (numpy array, BGR)
        save_path: Path to save the grid image
        
    Returns:
        Grid image (numpy array)
    """
    h, w = image.shape[:2]
    
    # Resize for grid (each cell)
    cell_w, cell_h = 320, 240
    
    weather_types = ['rain', 'fog', 'dust']
    intensities = ['light', 'moderate', 'severe']
    
    grid_h = cell_h * 4  # original + 3 intensity rows
    grid_w = cell_w * 4  # label + 3 weather columns
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark background
    
    # Original in top-left
    orig_resized = cv2.resize(image, (cell_w, cell_h))
    
    # Add labels and images
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header row: Original + weather type names
    cv2.putText(grid, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    grid[40:40+cell_h, 0:cell_w] = orig_resized
    
    for j, wt in enumerate(weather_types):
        cv2.putText(grid, wt.upper(), (cell_w*(j+1) + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Fill grid
    for i, intensity in enumerate(intensities):
        y_start = cell_h * (i + 1)
        cv2.putText(grid, intensity, (5, y_start + 20), font, 0.5, (200, 200, 200), 1)
        
        for j, wt in enumerate(weather_types):
            augmented = apply_weather(image, wt, intensity)
            augmented_resized = cv2.resize(augmented, (cell_w, cell_h))
            
            x_start = cell_w * (j + 1)
            grid[y_start:y_start+cell_h, x_start:x_start+cell_w] = augmented_resized
    
    if save_path:
        cv2.imwrite(str(save_path), grid)
    
    return grid


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthetic Weather Engine')
    parser.add_argument('--dataset-root', type=str, default='VisDrone2019-MOT-train')
    parser.add_argument('--output-dir', type=str, default='outputs/augmented')
    parser.add_argument('--sequences', type=int, default=5,
                        help='Number of sequences to augment')
    parser.add_argument('--weather', nargs='+', default=['rain', 'fog', 'dust'])
    parser.add_argument('--intensities', nargs='+', default=['light', 'moderate', 'severe'])
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    sequences_dir = dataset_root / 'sequences'
    annotations_dir = dataset_root / 'annotations'
    output_dir = Path(args.output_dir)
    
    # Get sequence names
    seq_names = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    selected_seqs = seq_names[:args.sequences]
    
    print(f"Augmenting {len(selected_seqs)} sequences with {args.weather} at {args.intensities}")
    
    # Generate comparison grid from first frame of first sequence
    first_frame = sorted((sequences_dir / selected_seqs[0]).glob('*.jpg'))[0]
    sample_img = cv2.imread(str(first_frame))
    grid_path = output_dir / 'weather_comparison_grid.jpg'
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    generate_comparison_grid(sample_img, grid_path)
    print(f"Saved comparison grid to {grid_path}")
    
    # Augment sequences
    for seq_name in tqdm(selected_seqs, desc="Sequences"):
        for weather_type in args.weather:
            for intensity in args.intensities:
                augment_sequence(
                    sequences_dir / seq_name,
                    annotations_dir / f'{seq_name}.txt',
                    output_dir,
                    weather_type,
                    intensity
                )
    
    print("Weather augmentation complete!")
