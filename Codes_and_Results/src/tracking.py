"""
Multi-Object Tracking Pipeline using BoxMOT
Supports ByteTrack and DeepSORT tracker variants.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Fix CUDA DLL path for Windows
def _fix_cuda_path():
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

_fix_cuda_path()

import torch


class TrackerWrapper:
    """
    Unified tracker wrapper supporting ByteTrack and DeepSORT via boxmot.
    """
    
    def __init__(self, tracker_type='bytetrack', device='cpu'):
        """
        Initialize tracker.
        
        Args:
            tracker_type: 'bytetrack' or 'deepsort'
            device: 'cpu' or 'cuda:0'
        """
        self.tracker_type = tracker_type
        self.device = device
        self.tracker = self._create_tracker()
    
    def _create_tracker(self):
        """Create the tracker instance based on type."""
        try:
            if self.tracker_type == 'bytetrack':
                from boxmot import ByteTrack
                return ByteTrack()
            elif self.tracker_type == 'deepsort':
                from boxmot import DeepSort
                return DeepSort(
                    model_weights=Path('osnet_x0_25_msmt17.pt'),
                    device=self.device,
                    fp16=False,
                )
            else:
                raise ValueError(f"Unknown tracker type: {self.tracker_type}")
        except ImportError as e:
            print(f"Warning: boxmot import failed ({e}). Using simple tracker fallback.")
            return SimpleTracker()
        except Exception as e:
            print(f"Warning: tracker init failed ({e}). Using simple tracker fallback.")
            return SimpleTracker()
    
    def update(self, detections, frame):
        """
        Update tracker with new detections.
        
        Args:
            detections: numpy array of shape (N, 6) -> [x1, y1, x2, y2, conf, class]
            frame: Current frame (numpy array, BGR)
            
        Returns:
            numpy array of shape (M, 7) -> [x1, y1, x2, y2, track_id, conf, class]
        """
        if len(detections) == 0:
            dets = np.empty((0, 6))
        else:
            dets = np.array(detections, dtype=np.float32)
        
        try:
            tracks = self.tracker.update(dets, frame)
            if tracks is not None and len(tracks) > 0:
                return np.array(tracks)
            return np.empty((0, 7))
        except Exception as e:
            print(f"Tracker update error: {e}")
            return np.empty((0, 7))


class SimpleTracker:
    """
    Simple IoU-based tracker fallback when boxmot is not available.
    Uses Hungarian algorithm for assignment.
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections, frame=None):
        """Update tracks with new detections."""
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age out old tracks
            to_remove = []
            for tid, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return np.empty((0, 7))
        
        dets = np.array(detections, dtype=np.float32)
        
        if len(self.tracks) == 0:
            # Initialize tracks
            results = []
            for det in dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det[:4], 'conf': det[4], 'cls': det[5],
                    'age': 0, 'hits': 1
                }
                results.append([*det[:4], tid, det[4], det[5]])
            return np.array(results)
        
        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
        
        iou_matrix = self._compute_iou_matrix(dets[:, :4], track_bboxes)
        
        # Greedy matching
        matched_dets = set()
        matched_tracks = set()
        matches = []
        
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            di, ti = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((di, ti))
            matched_dets.add(di)
            matched_tracks.add(ti)
            iou_matrix[di, :] = 0
            iou_matrix[:, ti] = 0
        
        results = []
        
        # Update matched tracks
        for di, ti in matches:
            tid = track_ids[ti]
            det = dets[di]
            self.tracks[tid]['bbox'] = det[:4]
            self.tracks[tid]['conf'] = det[4]
            self.tracks[tid]['cls'] = det[5]
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1
            
            if self.tracks[tid]['hits'] >= self.min_hits:
                results.append([*det[:4], tid, det[4], det[5]])
        
        # Create new tracks for unmatched detections
        for di in range(len(dets)):
            if di not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                det = dets[di]
                self.tracks[tid] = {
                    'bbox': det[:4], 'conf': det[4], 'cls': det[5],
                    'age': 0, 'hits': 1
                }
                results.append([*det[:4], tid, det[4], det[5]])
        
        # Age unmatched tracks
        to_remove = []
        for ti_idx, tid in enumerate(track_ids):
            if ti_idx not in matched_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]
        
        return np.array(results) if results else np.empty((0, 7))
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        n = len(boxes1)
        m = len(boxes2)
        iou_matrix = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])
        
        return iou_matrix
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def run_tracking_on_sequence(model_path, seq_dir, tracker_type='bytetrack',
                              conf_thresh=0.25, imgsz=640, device=None,
                              use_sahi=False, sahi_slice_size=320):
    """
    Run full detection + tracking pipeline on a video sequence.
    
    Args:
        model_path: Path to trained YOLO model
        seq_dir: Directory containing sequence frames
        tracker_type: 'bytetrack' or 'deepsort'
        conf_thresh: Detection confidence threshold
        imgsz: Input image size for detector
        device: Computation device
        use_sahi: Whether to use SAHI for detection
        sahi_slice_size: SAHI tile size
        
    Returns:
        dict: {frame_id: [(x1,y1,x2,y2, track_id, conf, class), ...]}
    """
    from ultralytics import YOLO
    
    seq_dir = Path(seq_dir)
    model = YOLO(model_path)
    
    # Initialize tracker
    tracker = TrackerWrapper(tracker_type, device or 'cpu')
    
    # SAHI model for sliced detection
    sahi_model = None
    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=str(model_path),
                confidence_threshold=conf_thresh,
                device=device or 'cpu'
            )
        except ImportError:
            print("SAHI not available, falling back to baseline detection")
            use_sahi = False
    
    frame_files = sorted(seq_dir.glob('*.jpg'))
    all_tracks = {}
    
    for frame_path in tqdm(frame_files, desc=f"Tracking {seq_dir.name}"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        frame_id = int(frame_path.stem)
        
        # Detect objects
        if use_sahi and sahi_model is not None:
            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                str(frame_path), sahi_model,
                slice_height=sahi_slice_size,
                slice_width=sahi_slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0
            )
            dets = []
            for pred in result.object_prediction_list:
                bbox = pred.bbox
                dets.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy,
                            pred.score.value, pred.category.id])
            dets = np.array(dets) if dets else np.empty((0, 6))
        else:
            predict_args = {
                'source': frame,
                'conf': conf_thresh,
                'imgsz': imgsz,
                'verbose': False,
                'save': False,
            }
            if device:
                predict_args['device'] = device
                
            results = model.predict(**predict_args)
            
            dets = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        dets.append([x1, y1, x2, y2, conf, cls])
            dets = np.array(dets) if dets else np.empty((0, 6))
        
        # Update tracker
        tracks = tracker.update(dets, frame)
        
        if len(tracks) > 0:
            all_tracks[frame_id] = tracks.tolist()
        else:
            all_tracks[frame_id] = []
    
    return all_tracks


def save_tracks_mot_format(tracks, output_path):
    """
    Save tracking results in MOT challenge format.
    
    Format: frame,id,x,y,w,h,conf,-1,-1,-1
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for frame_id in sorted(tracks.keys()):
            for track in tracks[frame_id]:
                if len(track) >= 6:
                    x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
                    tid = int(track[4])
                    conf = track[5] if len(track) > 5 else 1.0
                    cls_id = int(track[6]) if len(track) > 6 else -1
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls_id},-1,-1\n")


def run_tracking_experiment(model_path, dataset_root, output_dir,
                            sequence_names, tracker_type='bytetrack',
                            use_sahi=False, weather_type=None,
                            intensity=None, device=None):
    """
    Run tracking on multiple sequences and save results.
    
    Args:
        model_path: Path to YOLO model
        dataset_root: VisDrone dataset root
        output_dir: Output directory for tracks
        sequence_names: List of sequence names to process
        tracker_type: 'bytetrack' or 'deepsort'
        use_sahi: Whether to use SAHI
        weather_type: Weather condition (None for original)
        intensity: Weather intensity
        device: Computation device
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    sahi_tag = 'sahi' if use_sahi else 'baseline'
    weather_tag = f'{weather_type}_{intensity}' if weather_type else 'original'
    
    experiment_dir = output_dir / f'{tracker_type}_{sahi_tag}_{weather_tag}'
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    if weather_type and intensity:
        seq_base = dataset_root / 'outputs' / 'augmented' / f'{weather_type}_{intensity}' / 'sequences'
    else:
        seq_base = dataset_root / 'sequences'
    
    for seq_name in sequence_names:
        seq_dir = seq_base / seq_name
        if not seq_dir.exists():
            print(f"Sequence not found: {seq_dir}")
            continue
        
        tracks = run_tracking_on_sequence(
            model_path, seq_dir, tracker_type,
            use_sahi=use_sahi, device=device
        )
        
        save_tracks_mot_format(tracks, experiment_dir / seq_name / f'{seq_name}.txt')
    
    print(f"Tracking experiment saved to {experiment_dir}")
    return str(experiment_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Object Tracking Pipeline')
    parser.add_argument('--model', type=str, required=True, help='YOLO model path')
    parser.add_argument('--dataset-root', type=str, default='VisDrone2019-MOT-train')
    parser.add_argument('--output-dir', type=str, default='outputs/tracks')
    parser.add_argument('--sequences', nargs='+', default=None)
    parser.add_argument('--tracker', choices=['bytetrack', 'deepsort'], default='bytetrack')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    if args.sequences is None:
        seq_dir = dataset_root / 'sequences'
        args.sequences = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])[:5]
    
    run_tracking_experiment(
        model_path=args.model,
        dataset_root=str(dataset_root),
        output_dir=args.output_dir,
        sequence_names=args.sequences,
        tracker_type=args.tracker,
        use_sahi=args.sahi,
        device=args.device
    )
