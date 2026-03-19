"""
Detection Pipeline: YOLOv8 + SAHI Integration
Handles training, baseline inference, and SAHI-enhanced inference.
"""

import os
import sys
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Fix CUDA DLL path for Windows
def _fix_cuda_path():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(torch_lib)
            if torch_lib not in os.environ.get('PATH', ''):
                os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

_fix_cuda_path()


def create_yolo_config(output_dir, data_root):
    """
    Create YOLO dataset configuration YAML file.
    
    Args:
        output_dir: Where to save the config
        data_root: Root of the YOLO-formatted dataset
    """
    config_path = Path(output_dir) / 'visdrone.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(data_root).resolve()
    
    config = f"""# VisDrone2019 MOT Dataset - YOLO Format
path: {str(data_root)}
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
        f.write(config)
    
    print(f"YOLO config saved to {config_path}")
    return str(config_path)


def train_yolo(data_yaml, model_name='yolov8s.pt', epochs=50, imgsz=640,
               batch=16, project='runs/detect', name='visdrone_train',
               device=None):
    """
    Train/fine-tune YOLOv8 on the VisDrone dataset.
    
    Args:
        data_yaml: Path to dataset YAML config
        model_name: Base model to fine-tune
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory for results
        name: Experiment name
        device: Device ('cuda:0', 'cpu', etc.)
    """
    from ultralytics import YOLO
    
    model = YOLO(model_name)
    
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project,
        'name': name,
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'plots': True,
        'patience': 15,
        'workers': 0,
    }
    
    if device is not None:
        train_args['device'] = device
    
    results = model.train(**train_args)
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    
    return results


def detect_baseline(model_path, image_dir, output_dir, conf_thresh=0.25,
                    imgsz=640, device=None):
    """
    Run standard YOLOv8 inference (no SAHI).
    
    Args:
        model_path: Path to trained YOLO model
        image_dir: Directory containing images
        output_dir: Where to save detection results
        conf_thresh: Confidence threshold
        imgsz: Input image size
        device: Device for inference
        
    Returns:
        dict: {image_name: [(x1,y1,x2,y2, conf, class_id), ...]}
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_detections = {}
    image_files = sorted(image_dir.glob('*.jpg'))
    
    for img_path in tqdm(image_files, desc="YOLO Baseline Detection"):
        predict_args = {
            'source': str(img_path),
            'conf': conf_thresh,
            'imgsz': imgsz,
            'verbose': False,
            'save': False,
        }
        if device is not None:
            predict_args['device'] = device
            
        results = model.predict(**predict_args)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    detections.append((float(x1), float(y1), float(x2), float(y2), conf, cls))
        
        all_detections[img_path.name] = detections
    
    # Save detections
    det_path = output_dir / 'detections_baseline.json'
    with open(det_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"Baseline detections saved to {det_path}")
    return all_detections


def detect_sahi(model_path, image_dir, output_dir, conf_thresh=0.25,
                slice_size=320, overlap_ratio=0.2, device=None):
    """
    Run SAHI-enhanced YOLOv8 inference for small object detection.
    
    Args:
        model_path: Path to trained YOLO model
        image_dir: Directory containing images
        output_dir: Where to save detection results
        conf_thresh: Confidence threshold
        slice_size: Size of each tile
        overlap_ratio: Overlap between tiles
        device: Device for inference
        
    Returns:
        dict: {image_name: [(x1,y1,x2,y2, conf, class_id), ...]}
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf_thresh,
        device=device or 'cpu'
    )
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_detections = {}
    image_files = sorted(image_dir.glob('*.jpg'))
    
    for img_path in tqdm(image_files, desc="SAHI Detection"):
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0
        )
        
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            conf = pred.score.value
            cls = pred.category.id
            detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
        
        all_detections[img_path.name] = detections
    
    # Save detections
    det_path = output_dir / 'detections_sahi.json'
    with open(det_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"SAHI detections saved to {det_path}")
    return all_detections


def detections_to_mot_format(detections_dict, output_dir, sequence_name):
    """
    Convert detection results to MOT challenge format for tracking input.
    
    MOT format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    For detection input, id=-1.
    
    Args:
        detections_dict: {image_name: [(x1,y1,x2,y2,conf,cls), ...]}
        output_dir: Output directory
        sequence_name: Name of the sequence
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    det_file = output_dir / f'{sequence_name}.txt'
    
    with open(det_file, 'w') as f:
        for img_name, dets in sorted(detections_dict.items()):
            # Extract frame number from filename
            frame_num = int(Path(img_name).stem.split('_')[-1]) if '_' in img_name else int(Path(img_name).stem)
            
            for det in dets:
                x1, y1, x2, y2, conf, cls = det
                w = x2 - x1
                h = y2 - y1
                f.write(f"{frame_num},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")
    
    return str(det_file)


def compute_detection_metrics(detections, ground_truth, iou_threshold=0.5):
    """
    Compute detection metrics (precision, recall, mAP approximation).
    
    Args:
        detections: {image_name: [(x1,y1,x2,y2,conf,cls), ...]}
        ground_truth: {image_name: [(x1,y1,x2,y2,cls), ...]}
        iou_threshold: IoU threshold for matching
        
    Returns:
        dict with precision, recall, f1, tp, fp, fn counts
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_name in ground_truth:
        gt_boxes = ground_truth[img_name]
        det_boxes = detections.get(img_name, [])
        
        # Sort detections by confidence (highest first)
        det_boxes = sorted(det_boxes, key=lambda x: x[4], reverse=True)
        
        matched_gt = set()
        tp = 0
        fp = 0
        
        for det in det_boxes:
            dx1, dy1, dx2, dy2, conf, cls = det
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                gx1, gy1, gx2, gy2 = gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]
                
                # Compute IoU
                ix1 = max(dx1, gx1)
                iy1 = max(dy1, gy1)
                ix2 = min(dx2, gx2)
                iy2 = min(dy2, gy2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area_det = (dx2 - dx1) * (dy2 - dy1)
                    area_gt = (gx2 - gx1) * (gy2 - gy1)
                    union = area_det + area_gt - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Detection Pipeline')
    parser.add_argument('--mode', choices=['train', 'detect', 'sahi'],
                        required=True, help='Operation mode')
    parser.add_argument('--data-yaml', type=str, default='config/visdrone.yaml')
    parser.add_argument('--model', type=str, default='yolov8s.pt')
    parser.add_argument('--image-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs/detections')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_yolo(
            data_yaml=args.data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device
        )
    elif args.mode == 'detect':
        detect_baseline(
            model_path=args.model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            conf_thresh=args.conf,
            imgsz=args.imgsz,
            device=args.device
        )
    elif args.mode == 'sahi':
        detect_sahi(
            model_path=args.model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            conf_thresh=args.conf,
            device=args.device
        )
