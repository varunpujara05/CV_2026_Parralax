"""
Dataset Utilities for VisDrone2019-MOT
Handles parsing VisDrone annotations and converting to YOLO detection format.
"""

import os
import csv
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# VisDrone MOT annotation format:
# <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
# <score>,<object_category>,<truncation>,<occlusion>
#
# Categories: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
#             5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others

# Map VisDrone categories to YOLO class indices (skip 0=ignored, 11=others)
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

YOLO_CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]


def parse_visdrone_mot_annotation(txt_path):
    """
    Parse a VisDrone MOT annotation file.
    
    Args:
        txt_path: Path to the annotation .txt file
        
    Returns:
        dict: {frame_id: [(target_id, x, y, w, h, score, category, truncation, occlusion), ...]}
    """
    annotations = defaultdict(list)
    
    with open(txt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            frame_id = int(row[0])
            target_id = int(row[1])
            bbox_left = int(row[2])
            bbox_top = int(row[3])
            bbox_width = int(row[4])
            bbox_height = int(row[5])
            score = int(row[6])
            category = int(row[7])
            truncation = int(row[8])
            occlusion = int(row[9])
            
            annotations[frame_id].append({
                'target_id': target_id,
                'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                'score': score,
                'category': category,
                'truncation': truncation,
                'occlusion': occlusion
            })
    
    return annotations


def get_image_dimensions(image_path):
    """Get image width and height."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert VisDrone bbox (x, y, w, h) to YOLO format (cx, cy, w, h) normalized.
    
    Args:
        bbox: (x_left, y_top, width, height) in pixels
        img_w: Image width
        img_h: Image height
        
    Returns:
        (center_x, center_y, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox
    
    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return None
    
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    # Clamp normalized values
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    
    return cx, cy, nw, nh


def visdrone_mot_to_yolo_detection(dataset_root, output_dir, val_ratio=0.2, 
                                    sample_rate=5, min_bbox_area=100):
    """
    Convert VisDrone MOT dataset to YOLO detection format.
    
    We sample every Nth frame to avoid redundancy (consecutive MOT frames are very similar).
    
    Args:
        dataset_root: Path to VisDrone2019-MOT-train/
        output_dir: Output directory for YOLO format dataset
        val_ratio: Fraction of sequences to use for validation
        sample_rate: Sample every Nth frame (reduce redundancy)
        min_bbox_area: Minimum bbox area in pixels to include
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    sequences_dir = dataset_root / 'sequences'
    annotations_dir = dataset_root / 'annotations'
    
    # Get all sequence names
    sequence_names = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    # Split sequences into train/val
    train_seqs, val_seqs = train_test_split(
        sequence_names, test_size=val_ratio, random_state=42
    )
    
    print(f"Total sequences: {len(sequence_names)}")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences: {len(val_seqs)}")
    
    # Create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {'train': {'images': 0, 'objects': 0}, 'val': {'images': 0, 'objects': 0}}
    class_counts = defaultdict(int)
    
    for seq_name in tqdm(sequence_names, desc="Converting sequences"):
        split = 'train' if seq_name in train_seqs else 'val'
        
        # Parse annotations
        ann_path = annotations_dir / f'{seq_name}.txt'
        if not ann_path.exists():
            print(f"Warning: Annotation file not found for {seq_name}")
            continue
            
        annotations = parse_visdrone_mot_annotation(ann_path)
        
        # Get frame files
        seq_dir = sequences_dir / seq_name
        frame_files = sorted(seq_dir.glob('*.jpg'))
        
        if not frame_files:
            continue
        
        # Get image dimensions from first frame
        img_w, img_h = get_image_dimensions(frame_files[0])
        
        # Process frames with sampling
        for idx, frame_path in enumerate(frame_files):
            if idx % sample_rate != 0:
                continue
            
            frame_id = idx + 1  # VisDrone frames are 1-indexed
            frame_annotations = annotations.get(frame_id, [])
            
            # Convert annotations to YOLO format
            yolo_labels = []
            for ann in frame_annotations:
                cat = ann['category']
                if cat not in VISDRONE_TO_YOLO:
                    continue  # Skip ignored regions and 'others'
                
                # Skip low-score annotations (score=0 means not considered)
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]
                if area < min_bbox_area:
                    continue
                
                yolo_bbox = convert_bbox_to_yolo(bbox, img_w, img_h)
                if yolo_bbox is None:
                    continue
                
                yolo_class = VISDRONE_TO_YOLO[cat]
                cx, cy, nw, nh = yolo_bbox
                yolo_labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                class_counts[YOLO_CLASS_NAMES[yolo_class]] += 1
            
            # Create unique filename
            out_name = f"{seq_name}_{frame_path.stem}"
            
            # Copy image
            dst_img = output_dir / 'images' / split / f'{out_name}.jpg'
            shutil.copy2(frame_path, dst_img)
            
            # Write label file
            dst_label = output_dir / 'labels' / split / f'{out_name}.txt'
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            stats[split]['images'] += 1
            stats[split]['objects'] += len(yolo_labels)
    
    # Print statistics
    print("\n=== Dataset Conversion Statistics ===")
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        print(f"  Images: {stats[split]['images']}")
        print(f"  Objects: {stats[split]['objects']}")
        if stats[split]['images'] > 0:
            print(f"  Avg objects/image: {stats[split]['objects'] / stats[split]['images']:.1f}")
    
    print("\nClass distribution:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {count}")
    
    return stats


def get_sequence_info(dataset_root):
    """Get information about all sequences in the dataset."""
    dataset_root = Path(dataset_root)
    sequences_dir = dataset_root / 'sequences'
    annotations_dir = dataset_root / 'annotations'
    
    info = []
    for seq_dir in sorted(sequences_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        
        frames = list(seq_dir.glob('*.jpg'))
        ann_path = annotations_dir / f'{seq_dir.name}.txt'
        
        ann_data = parse_visdrone_mot_annotation(ann_path)
        unique_ids = set()
        total_objects = 0
        categories = defaultdict(int)
        
        for frame_id, anns in ann_data.items():
            for ann in anns:
                if ann['category'] in VISDRONE_TO_YOLO:
                    unique_ids.add(ann['target_id'])
                    total_objects += 1
                    categories[ann['category']] += 1
        
        info.append({
            'name': seq_dir.name,
            'num_frames': len(frames),
            'num_unique_ids': len(unique_ids),
            'total_annotations': total_objects,
            'categories': dict(categories)
        })
    
    return info


def create_mot_ground_truth(dataset_root, output_dir, sequence_names=None):
    """
    Create MOT challenge format ground truth files for evaluation.
    
    Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    
    Args:
        dataset_root: Path to VisDrone2019-MOT-train/
        output_dir: Output directory for GT files
        sequence_names: List of sequence names to process (None = all)
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    sequences_dir = dataset_root / 'sequences'
    annotations_dir = dataset_root / 'annotations'
    
    if sequence_names is None:
        sequence_names = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    for seq_name in tqdm(sequence_names, desc="Creating MOT GT"):
        ann_path = annotations_dir / f'{seq_name}.txt'
        annotations = parse_visdrone_mot_annotation(ann_path)
        
        gt_dir = output_dir / seq_name / 'gt'
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        with open(gt_dir / 'gt.txt', 'w') as f:
            for frame_id in sorted(annotations.keys()):
                for ann in annotations[frame_id]:
                    if ann['category'] not in VISDRONE_TO_YOLO:
                        continue
                    x, y, w, h = ann['bbox']
                    tid = ann['target_id']
                    # MOT format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
                    f.write(f"{frame_id},{tid},{x},{y},{w},{h},1,{ann['category']},1.0\n")
        
        # Create seqinfo.ini
        seq_dir = sequences_dir / seq_name
        frames = sorted(seq_dir.glob('*.jpg'))
        if frames:
            img = cv2.imread(str(frames[0]))
            h, w = img.shape[:2]
            seq_len = len(frames)
            
            with open(output_dir / seq_name / 'seqinfo.ini', 'w') as f:
                f.write(f"[Sequence]\n")
                f.write(f"name={seq_name}\n")
                f.write(f"imDir=img1\n")
                f.write(f"frameRate=30\n")
                f.write(f"seqLength={seq_len}\n")
                f.write(f"imWidth={w}\n")
                f.write(f"imHeight={h}\n")
                f.write(f"imExt=.jpg\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VisDrone Dataset Utilities')
    parser.add_argument('--dataset-root', type=str, 
                        default='VisDrone2019-MOT-train',
                        help='Path to VisDrone2019-MOT-train directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/visdrone_yolo',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--sample-rate', type=int, default=5,
                        help='Sample every Nth frame')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Print dataset info
    print("=== VisDrone Dataset Info ===")
    info = get_sequence_info(args.dataset_root)
    for seq in info[:5]:
        print(f"  {seq['name']}: {seq['num_frames']} frames, "
              f"{seq['num_unique_ids']} unique IDs, "
              f"{seq['total_annotations']} annotations")
    print(f"  ... ({len(info)} sequences total)")
    
    # Convert to YOLO format
    print("\n=== Converting to YOLO Format ===")
    visdrone_mot_to_yolo_detection(
        args.dataset_root, args.output_dir,
        val_ratio=args.val_ratio,
        sample_rate=args.sample_rate
    )
