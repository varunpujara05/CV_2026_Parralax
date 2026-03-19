"""
Visualization Module for UAV MOT Pipeline
Generates all plots, comparison grids, trajectory overlays, and video outputs.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import colorsys
import random
import os


# ============================================================
#  COLOR UTILITIES
# ============================================================

def generate_colors(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        h = i / n
        s = 0.8 + random.random() * 0.2
        v = 0.8 + random.random() * 0.2
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    random.shuffle(colors)
    return colors


def get_track_color(track_id, color_map=None):
    """Get a consistent color for a track ID."""
    if color_map is None:
        color_map = {}
    if track_id not in color_map:
        h = (track_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        color_map[track_id] = (int(b * 255), int(g * 255), int(r * 255))
    return color_map[track_id]


# ============================================================
#  DETECTION ANALYSIS PLOTS
# ============================================================

def plot_map_vs_weather(results_df, save_path='outputs/plots/map_vs_weather.png'):
    """
    Plot mAP (or Precision) vs weather type and intensity.
    
    Args:
        results_df: DataFrame with columns [Experiment, MOTA, Precision, Recall, ...]
        save_path: Where to save the plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detection Performance vs Weather Conditions', fontsize=16, fontweight='bold')
    
    metrics = ['Precision', 'Recall', 'MOTA']
    colors_list = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric in results_df.columns:
            data = results_df[['Experiment', metric]].copy()
            bars = ax.barh(data['Experiment'], data[metric], 
                          color=[colors_list[i % len(colors_list)] for i in range(len(data))],
                          edgecolor='white', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, data[metric]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=9)
            
            ax.set_xlabel(f'{metric} (%)')
            ax.set_title(metric, fontsize=13, fontweight='bold')
            ax.set_xlim(0, 105)
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall_curves(results_by_condition, save_path='outputs/plots/pr_curves.png'):
    """
    Plot Precision-Recall curves per weather condition.
    
    Args:
        results_by_condition: dict of {condition: {'precision': [...], 'recall': [...]}}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1976D2', '#D32F2F', '#388E3C', '#F57C00', '#7B1FA2',
              '#00796B', '#C2185B', '#FBC02D', '#455A64']
    
    for idx, (condition, data) in enumerate(results_by_condition.items()):
        color = colors[idx % len(colors)]
        precision = data.get('precision', [0])
        recall = data.get('recall', [0])
        
        ax.plot(recall, precision, 'o-', color=color, label=condition, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Recall (%)', fontsize=12)
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Precision vs Recall by Weather Condition', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_object_size_vs_accuracy(gt_data, pred_data, save_path='outputs/plots/size_vs_accuracy.png'):
    """
    Plot detection accuracy vs object size (bbox area).
    
    Groups objects into size buckets and shows per-bucket detection rate.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define size buckets (area in pixels)
    buckets = {
        'Tiny\n(<500px²)': (0, 500),
        'Small\n(500-2K)': (500, 2000),
        'Medium\n(2K-10K)': (2000, 10000),
        'Large\n(10K-50K)': (10000, 50000),
        'Very Large\n(>50K)': (50000, float('inf'))
    }
    
    bucket_counts = {name: {'total': 0, 'detected': 0} for name in buckets}
    
    for frame_id in gt_data:
        gt_objs = gt_data[frame_id]
        pred_objs = pred_data.get(frame_id, [])
        
        for gt_obj in gt_objs:
            x, y, w, h = gt_obj['bbox']
            area = w * h
            
            # Find bucket
            for bname, (low, high) in buckets.items():
                if low <= area < high:
                    bucket_counts[bname]['total'] += 1
                    
                    # Check if detected (any prediction with IoU > 0.5)
                    detected = False
                    for pred_obj in pred_objs:
                        from src.evaluation import compute_iou
                        iou = compute_iou(gt_obj['bbox'], pred_obj['bbox'])
                        if iou >= 0.5:
                            detected = True
                            break
                    
                    if detected:
                        bucket_counts[bname]['detected'] += 1
                    break
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Object Size vs Detection Performance', fontsize=14, fontweight='bold')
    
    names = list(buckets.keys())
    totals = [bucket_counts[n]['total'] for n in names]
    detected = [bucket_counts[n]['detected'] for n in names]
    rates = [d/t*100 if t > 0 else 0 for d, t in zip(detected, totals)]
    
    # Bar chart of detection rates
    bars = ax1.bar(names, rates, color=['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0'],
                   edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Detection Rate by Object Size')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=9)
    
    # Distribution of object sizes
    ax2.bar(names, totals, color='#607D8B', alpha=0.7, edgecolor='white')
    ax2.set_ylabel('Count')
    ax2.set_title('Object Size Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
#  TRACKING ANALYSIS PLOTS
# ============================================================

def plot_tracking_vs_weather(results_df, save_path='outputs/plots/tracking_vs_weather.png'):
    """
    Plot MOTA/HOTA/IDF1 across weather conditions.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Tracking Performance vs Weather Severity', fontsize=16, fontweight='bold')
    
    metrics = ['MOTA', 'MOTP', 'IDF1']
    palettes = ['Blues_d', 'Greens_d', 'Oranges_d']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric in results_df.columns:
            data = results_df[['Experiment', metric]].sort_values(metric, ascending=True)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
            bars = ax.barh(data['Experiment'], data[metric], color=colors,
                          edgecolor='white', linewidth=0.5)
            
            for bar, val in zip(bars, data[metric]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=9)
            
            ax.set_xlabel(f'{metric} (%)')
            ax.set_title(metric, fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_id_switches_analysis(results_df, save_path='outputs/plots/id_switches.png'):
    """
    Plot ID switches across experiments.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ID Switch Analysis', fontsize=14, fontweight='bold')
    
    if 'ID_Switches' in results_df.columns:
        data = results_df[['Experiment', 'ID_Switches']].sort_values('ID_Switches')
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(data)))
        ax1.barh(data['Experiment'], data['ID_Switches'], color=colors,
                edgecolor='white', linewidth=0.5)
        ax1.set_xlabel('ID Switches')
        ax1.set_title('ID Switches by Condition', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    
    if 'FN' in results_df.columns:
        fn_cols = ['Experiment', 'FN']
        if 'FP' in results_df.columns:
            fn_cols.append('FP')
        data = results_df[fn_cols].sort_values('FN')
        
        x = np.arange(len(data))
        width = 0.35
        ax2.barh(x - width/2, data['FN'], width, label='False Negatives', 
                color='#F44336', alpha=0.8)
        if 'FP' in data.columns:
            ax2.barh(x + width/2, data['FP'], width, label='False Positives',
                    color='#FF9800', alpha=0.8)
        ax2.set_yticks(x)
        ax2.set_yticklabels(data['Experiment'])
        ax2.set_xlabel('Count')
        ax2.set_title('FN Analysis by Condition', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
#  VISUAL OUTPUTS
# ============================================================

def draw_detection_boxes(frame, detections, class_names=None, color=(0, 255, 0)):
    """
    Draw detection bounding boxes on a frame.
    
    Args:
        frame: Image (BGR numpy array)
        detections: List of (x1,y1,x2,y2,conf,cls) or (x,y,w,h,conf,cls)
        class_names: Optional list of class name strings
        color: Default BGR color
    """
    result = frame.copy()
    
    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
        else:
            continue
        
        # Draw box
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Label
        label = f'{class_names[cls] if class_names and cls < len(class_names) else cls}: {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(result, (int(x1), int(y1)-th-6), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(result, label, (int(x1), int(y1)-4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result


def draw_tracking_boxes(frame, tracks, color_map=None, trail_history=None):
    """
    Draw tracking boxes with ID labels and optional trail.
    
    Args:
        frame: Image (BGR numpy array)
        tracks: List of [x1,y1,x2,y2,track_id,conf,cls]
        color_map: Dict mapping track_id to color
        trail_history: Dict mapping track_id to list of center points
    """
    if color_map is None:
        color_map = {}
    
    result = frame.copy()
    
    for track in tracks:
        if len(track) < 5:
            continue
        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
        track_id = int(track[4])
        conf = track[5] if len(track) > 5 else 1.0
        
        color = get_track_color(track_id, color_map)
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID label
        label = f'ID:{track_id}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(result, label, (x1+2, y1-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw trail
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if trail_history is not None:
            if track_id not in trail_history:
                trail_history[track_id] = []
            trail_history[track_id].append((cx, cy))
            
            # Keep last 30 points
            trail_history[track_id] = trail_history[track_id][-30:]
            
            # Draw trail
            points = trail_history[track_id]
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(alpha * 3))
                cv2.line(result, points[i-1], points[i], color, thickness, cv2.LINE_AA)
    
    return result


def create_comparison_frames(original_dir, augmented_dirs, frame_idx=0,
                              save_path='outputs/plots/weather_comparison.png'):
    """
    Create side-by-side comparison of original vs weather-augmented frames.
    
    Args:
        original_dir: Path to original sequence directory
        augmented_dirs: Dict of {label: path_to_augmented_seq}
        frame_idx: Which frame to compare
        save_path: Where to save
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    original_dir = Path(original_dir)
    frames = sorted(original_dir.glob('*.jpg'))
    
    if frame_idx >= len(frames):
        frame_idx = 0
    
    original = cv2.imread(str(frames[frame_idx]))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    n_cols = 1 + len(augmented_dirs)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for idx, (label, aug_dir) in enumerate(augmented_dirs.items()):
        aug_frames = sorted(Path(aug_dir).glob('*.jpg'))
        if frame_idx < len(aug_frames):
            aug_img = cv2.imread(str(aug_frames[frame_idx]))
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            axes[idx+1].imshow(aug_img)
        axes[idx+1].set_title(label, fontsize=12, fontweight='bold')
        axes[idx+1].axis('off')
    
    plt.suptitle('Weather Augmentation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def render_tracked_video(seq_dir, tracks_data, output_path, fps=30, 
                          max_frames=None, show_trails=True):
    """
    Render a video with tracking overlays.
    
    Args:
        seq_dir: Path to frame directory
        tracks_data: {frame_id: [(x1,y1,x2,y2,track_id,conf,cls), ...]}
        output_path: Output video path
        fps: Frames per second
        max_frames: Maximum frames to render (None = all)
        show_trails: Whether to show tracking trails
    """
    seq_dir = Path(seq_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(seq_dir.glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    if not frame_files:
        print("No frames found!")
        return
    
    # Get frame dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    color_map = {}
    trail_history = {} if show_trails else None
    
    for frame_path in tqdm(frame_files, desc="Rendering video"):
        frame = cv2.imread(str(frame_path))
        frame_id = int(frame_path.stem)
        
        frame_tracks = tracks_data.get(frame_id, [])
        
        # Draw tracking visualization
        frame = draw_tracking_boxes(frame, frame_tracks, color_map, trail_history)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_id}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Objects: {len(frame_tracks)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"Video saved: {output_path}")


def plot_ablation_study(results_df, save_path='outputs/plots/ablation_study.png'):
    """
    Plot ablation study comparing SAHI vs no-SAHI and different trackers.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study Results', fontsize=14, fontweight='bold')
    
    metrics = ['MOTA', 'MOTP', 'IDF1']
    
    # SAHI comparison
    ax = axes[0]
    sahi_rows = results_df[results_df['Experiment'].str.contains('sahi', case=False)]
    baseline_rows = results_df[results_df['Experiment'].str.contains('baseline', case=False)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    if not baseline_rows.empty and not sahi_rows.empty:
        baseline_vals = [baseline_rows.iloc[0].get(m, 0) for m in metrics]
        sahi_vals = [sahi_rows.iloc[0].get(m, 0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                       color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, sahi_vals, width, label='+ SAHI',
                       color='#4CAF50', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score (%)')
        ax.set_title('SAHI Effect', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Tracker comparison
    ax = axes[1]
    bytetrack_rows = results_df[results_df['Experiment'].str.contains('bytetrack', case=False)]
    deepsort_rows = results_df[results_df['Experiment'].str.contains('deepsort', case=False)]
    
    if not bytetrack_rows.empty and not deepsort_rows.empty:
        bt_vals = [bytetrack_rows.iloc[0].get(m, 0) for m in metrics]
        ds_vals = [deepsort_rows.iloc[0].get(m, 0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, bt_vals, width, label='ByteTrack',
                       color='#FF9800', alpha=0.8)
        bars2 = ax.bar(x + width/2, ds_vals, width, label='DeepSORT',
                       color='#9C27B0', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score (%)')
        ax.set_title('Tracker Comparison', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comprehensive_dashboard(results_df, save_path='outputs/plots/dashboard.png'):
    """
    Create a comprehensive dashboard with all key metrics.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('UAV Multi-Object Tracking — Comprehensive Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    metrics = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall', 'ID_Switches']
    colors_list = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#00796B', '#D32F2F']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        if metric in results_df.columns:
            data = results_df[['Experiment', metric]].sort_values(metric, ascending=(metric == 'ID_Switches'))
            
            color = colors_list[idx]
            bars = ax.barh(data['Experiment'], data[metric], color=color, alpha=0.8,
                          edgecolor='white', linewidth=0.5)
            
            for bar, val in zip(bars, data[metric]):
                suffix = '%' if metric != 'ID_Switches' else ''
                ax.text(bar.get_width() + max(data[metric])*0.02, 
                       bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}{suffix}', va='center', fontsize=8)
            
            ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(axis='y', labelsize=8)
    
    # Summary table in bottom row
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    display_cols = [c for c in ['Experiment', 'MOTA', 'HOTA', 'IDF1', 'Precision', 'Recall', 'ID_Switches', 'FN'] 
                    if c in results_df.columns]
    if display_cols:
        table = ax_table.table(
            cellText=results_df[display_cols].values,
            colLabels=display_cols,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        
        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#37474F')
                cell.set_text_props(color='white', fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    print("Visualization module loaded. Import and call functions as needed.")
