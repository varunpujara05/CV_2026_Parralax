"""
Step 9: Enhanced Tracking Video Generation
Renders detailed tracking videos with:
- Color-coded bounding boxes per track ID
- Class labels (pedestrian, car, etc.) above each box
- Track ID labels
- Movement trails
- ID switch counter overlay
- Frame-by-frame statistics
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

import cv2
import numpy as np
import colorsys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# VisDrone class names
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tri', 'bus', 'motor'
]

# Class colors (BGR) - vivid, distinct
CLASS_COLORS = {
    0: (0, 255, 0),      # pedestrian - green
    1: (0, 200, 0),      # people - dark green
    2: (255, 165, 0),    # bicycle - orange
    3: (255, 0, 0),      # car - blue
    4: (255, 100, 100),  # van - light blue
    5: (0, 0, 255),      # truck - red
    6: (0, 255, 255),    # tricycle - yellow
    7: (128, 0, 128),    # awning-tricycle - purple
    8: (255, 0, 255),    # bus - magenta
    9: (0, 128, 255),    # motor - orange-ish
}


def get_track_color(track_id):
    """Generate a unique, vivid color for a track ID."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_enhanced_frame(frame, tracks, frame_id, trail_history, prev_ids,
                         id_switch_count, total_objects, condition_label=''):
    """
    Draw enhanced tracking visualization on a single frame.
    
    Args:
        frame: BGR image
        tracks: list of [x1, y1, x2, y2, track_id, conf, class_id]
        frame_id: current frame number
        trail_history: dict tracking center point history per ID
        prev_ids: set of track IDs from previous frame
        id_switch_count: cumulative ID switch counter
        total_objects: running total of unique objects
        condition_label: e.g. 'Original', 'Rain (Severe)'
    
    Returns:
        annotated frame, updated trail_history, current_ids set, 
        updated id_switch_count
    """
    result = frame.copy()
    h, w = result.shape[:2]
    current_ids = set()
    
    for track in tracks:
        if len(track) < 5:
            continue
        
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        track_id = int(track[4])
        conf = track[5] if len(track) > 5 else 0.0
        cls_id = int(track[6]) if len(track) > 6 else -1
        
        current_ids.add(track_id)
        
        # Get color based on track ID (consistent across frames)
        color = get_track_color(track_id)
        
        # Draw bounding box (thicker for larger objects)
        box_area = (x2 - x1) * (y2 - y1)
        thickness = 3 if box_area > 5000 else 2
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Class label + Track ID label above box
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f'cls{cls_id}'
        label = f'ID:{track_id} {cls_name} {conf:.2f}'
        
        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        label_y = max(y1 - 4, th + 8)
        cv2.rectangle(result, (x1, label_y - th - 6), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(result, label, (x1 + 2, label_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Center point for trail
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(result, (cx, cy), 3, color, -1)
        
        # Trail
        if track_id not in trail_history:
            trail_history[track_id] = []
        trail_history[track_id].append((cx, cy))
        trail_history[track_id] = trail_history[track_id][-40:]  # keep last 40 pts
        
        points = trail_history[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            t = max(1, int(alpha * 3))
            trail_color = tuple(int(c * alpha) for c in color)
            cv2.line(result, points[i-1], points[i], trail_color, t, cv2.LINE_AA)
    
    # Detect ID switches (IDs that disappeared and new ones appeared)
    if prev_ids:
        lost_ids = prev_ids - current_ids
        new_ids = current_ids - prev_ids
        # Simple heuristic: ID switches ~ min(lost, new)
        switches = min(len(lost_ids), len(new_ids))
        id_switch_count += switches
    
    # --- HUD Overlay ---
    # Dark semi-transparent header bar
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Frame info
    cv2.putText(result, f'Frame: {frame_id}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f'Objects: {len(tracks)}', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    
    # ID switch counter
    cv2.putText(result, f'ID Switches: {id_switch_count}', (250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
    
    # Condition label
    if condition_label:
        cv2.putText(result, condition_label, (250, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1, cv2.LINE_AA)
    
    # Unique objects count
    cv2.putText(result, f'Unique IDs: {total_objects}', (500, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    
    return result, trail_history, current_ids, id_switch_count


def render_enhanced_video(seq_dir, track_file, output_path, condition_label='',
                           fps=15, max_frames=None):
    """
    Render an enhanced tracking video.
    
    Args:
        seq_dir: path to frame images
        track_file: MOT format track file
        output_path: output .mp4 path
        condition_label: display label
        fps: video FPS
        max_frames: limit frames (None = all)
    """
    seq_dir = Path(seq_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load tracks (MOT format: frame,id,x,y,w,h,conf,-1,-1,-1)
    tracks_by_frame = defaultdict(list)
    if Path(track_file).exists():
        with open(track_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    frame_id = int(parts[0])
                    tid = int(parts[1])
                    x, y, w_box, h_box = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6])
                    # Convert x,y,w,h to x1,y1,x2,y2
                    x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                    # Try to get class from remaining fields (may not exist)
                    cls_id = int(float(parts[7])) if len(parts) > 7 and parts[7] != '-1' else 0
                    tracks_by_frame[frame_id].append([x1, y1, x2, y2, tid, conf, cls_id])
    
    frame_files = sorted(seq_dir.glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    if not frame_files:
        print(f"  No frames found in {seq_dir}")
        return
    
    # Get frame dimensions
    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    trail_history = {}
    prev_ids = set()
    id_switch_count = 0
    all_seen_ids = set()
    
    for frame_path in tqdm(frame_files, desc=f"  Rendering {output_path.stem}"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        frame_id = int(frame_path.stem)
        frame_tracks = tracks_by_frame.get(frame_id, [])
        
        # Track all unique IDs
        for t in frame_tracks:
            if len(t) >= 5:
                all_seen_ids.add(int(t[4]))
        
        result, trail_history, current_ids, id_switch_count = draw_enhanced_frame(
            frame, frame_tracks, frame_id, trail_history, prev_ids,
            id_switch_count, len(all_seen_ids), condition_label
        )
        
        prev_ids = current_ids
        writer.write(result)
    
    writer.release()
    print(f"  Video saved: {output_path} ({len(frame_files)} frames, {id_switch_count} ID switches)")


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tracks_dir = project_root / 'outputs' / 'tracks'
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    videos_dir = project_root / 'outputs' / 'videos'
    
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 9: ENHANCED TRACKING VIDEOS")
    print("=" * 60)
    
    condition_labels = {
        'original': 'Original',
        'rain_light': 'Rain (Light)',
        'rain_moderate': 'Rain (Moderate)',
        'rain_severe': 'Rain (Severe)',
        'fog_light': 'Fog (Light)',
        'fog_moderate': 'Fog (Moderate)',
        'fog_severe': 'Fog (Severe)',
        'dust_light': 'Dust (Light)',
        'dust_moderate': 'Dust (Moderate)',
        'dust_severe': 'Dust (Severe)',
    }
    
    for exp_dir in sorted(tracks_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        print(f"\n--- {exp_name} ---")
        
        # Determine condition for label
        label = 'Original'
        for key, lbl in condition_labels.items():
            if key in exp_name:
                label = lbl
                break
        
        # Add detection mode
        if 'sahi' in exp_name:
            label += ' + SAHI'
        else:
            label += ' + Baseline'
        
        for seq_dir_track in sorted(exp_dir.iterdir()):
            if not seq_dir_track.is_dir():
                continue
            
            seq_name = seq_dir_track.name
            track_file = seq_dir_track / f'{seq_name}.txt'
            
            if not track_file.exists():
                continue
            
            # Find frame directory
            if 'original' in exp_name:
                frames_dir = sequences_dir / seq_name
            else:
                # Extract weather_intensity from experiment name
                found = False
                for weather in ['rain', 'fog', 'dust']:
                    for intensity in ['light', 'moderate', 'severe']:
                        if weather in exp_name and intensity in exp_name:
                            frames_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
                            found = True
                            break
                    if found:
                        break
                if not found:
                    frames_dir = sequences_dir / seq_name
            
            if not frames_dir.exists():
                print(f"  Frames not found: {frames_dir}")
                continue
            
            video_path = videos_dir / f'{exp_name}_{seq_name}.mp4'
            render_enhanced_video(
                str(frames_dir),
                str(track_file),
                str(video_path),
                condition_label=label,
                fps=15,
                max_frames=200  # First 200 frames for manageable size
            )
    
    print("\n" + "=" * 60)
    print("ENHANCED VIDEO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nVideos saved to: {videos_dir}")


if __name__ == '__main__':
    main()
