"""
Step 7: Visualization
Generate all plots, comparison images, and tracking videos.
"""

import sys
import os
import json
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    plot_map_vs_weather,
    plot_precision_recall_curves,
    plot_tracking_vs_weather,
    plot_id_switches_analysis,
    plot_ablation_study,
    plot_comprehensive_dashboard,
    create_comparison_frames,
    render_tracked_video,
    draw_tracking_boxes
)
from src.evaluation import load_mot_file
from pathlib import Path
import pandas as pd


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    eval_dir = project_root / 'outputs' / 'eval_results'
    plots_dir = project_root / 'outputs' / 'plots'
    videos_dir = project_root / 'outputs' / 'videos'
    tracks_dir = project_root / 'outputs' / 'tracks'
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    augmented_dir = project_root / 'outputs' / 'augmented'
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 7: VISUALIZATION")
    print("=" * 60)
    
    # 1. Load comparison data
    comparison_path = eval_dir / 'comparison.csv'
    if comparison_path.exists():
        results_df = pd.read_csv(comparison_path)
        print(f"\nLoaded {len(results_df)} experiment results")
        
        # Detection performance plots
        print("\n--- Generating Detection Plots ---")
        plot_map_vs_weather(results_df, str(plots_dir / 'map_vs_weather.png'))
        
        # Tracking performance plots
        print("\n--- Generating Tracking Plots ---")
        plot_tracking_vs_weather(results_df, str(plots_dir / 'tracking_vs_weather.png'))
        
        # ID switch analysis
        plot_id_switches_analysis(results_df, str(plots_dir / 'id_switches.png'))
        
        # Ablation study
        plot_ablation_study(results_df, str(plots_dir / 'ablation_study.png'))
        
        # Comprehensive dashboard
        plot_comprehensive_dashboard(results_df, str(plots_dir / 'dashboard.png'))
        
        # Precision-Recall curves
        pr_data = {}
        for _, row in results_df.iterrows():
            pr_data[row['Experiment']] = {
                'precision': [row.get('Precision', 0)],
                'recall': [row.get('Recall', 0)]
            }
        plot_precision_recall_curves(pr_data, str(plots_dir / 'pr_curves.png'))
    else:
        print("No evaluation results found. Using sample data for visualization demo...")
        
        # Create demo plots with sample data
        sample_data = pd.DataFrame({
            'Experiment': ['Original', 'Rain_Severe', 'Fog_Severe', 'Dust_Severe'],
            'MOTA': [45.2, 32.1, 28.5, 35.8],
            'HOTA': [38.5, 26.3, 22.1, 30.2],
            'IDF1': [52.1, 38.4, 34.2, 42.5],
            'Precision': [72.5, 58.3, 52.1, 62.4],
            'Recall': [61.3, 48.2, 42.5, 53.1],
            'ID_Switches': [125, 287, 342, 198],
            'FN': [1250, 2180, 2530, 1850],
            'FP': [520, 890, 1020, 720],
        })
        
        results_df = sample_data
        
        plot_map_vs_weather(sample_data, str(plots_dir / 'map_vs_weather.png'))
        plot_tracking_vs_weather(sample_data, str(plots_dir / 'tracking_vs_weather.png'))
        plot_id_switches_analysis(sample_data, str(plots_dir / 'id_switches.png'))
        plot_ablation_study(sample_data, str(plots_dir / 'ablation_study.png'))
        plot_comprehensive_dashboard(sample_data, str(plots_dir / 'dashboard.png'))
    
    # 2. Weather comparison images
    print("\n--- Generating Weather Comparisons ---")
    sequences_dir = dataset_root / 'sequences'
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    
    if all_seqs:
        seq_name = all_seqs[0]
        original_dir = sequences_dir / seq_name
        
        augmented_dirs = {}
        for weather in ['rain', 'fog', 'dust']:
            aug_dir = augmented_dir / f'{weather}_severe' / 'sequences' / seq_name
            if aug_dir.exists():
                augmented_dirs[f'{weather.title()} (Severe)'] = str(aug_dir)
        
        if augmented_dirs:
            create_comparison_frames(
                str(original_dir),
                augmented_dirs,
                frame_idx=50,
                save_path=str(plots_dir / 'weather_comparison.png')
            )
    
    # 3. Render tracking videos
    print("\n--- Rendering Tracking Videos ---")
    if tracks_dir.exists():
        for exp_dir in sorted(tracks_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            # Process first sequence only for video
            for seq_dir in sorted(exp_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                
                seq_name = seq_dir.name
                track_file = seq_dir / f'{seq_name}.txt'
                
                if not track_file.exists():
                    continue
                
                # Load tracks
                tracks_data = load_mot_file(str(track_file))
                
                # Find corresponding frames
                # Check if this is a weather-augmented sequence
                exp_name = exp_dir.name
                if 'original' in exp_name:
                    frames_dir = sequences_dir / seq_name
                else:
                    # Try to find augmented frames
                    parts = exp_name.split('_')
                    for weather in ['rain', 'fog', 'dust']:
                        for intensity in ['light', 'moderate', 'severe']:
                            if weather in exp_name and intensity in exp_name:
                                frames_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
                                break
                    else:
                        frames_dir = sequences_dir / seq_name
                
                if frames_dir.exists():
                    video_path = videos_dir / f'{exp_name}_{seq_name}.mp4'
                    print(f"  Rendering: {video_path.name}")
                    render_tracked_video(
                        str(frames_dir),
                        tracks_data,
                        str(video_path),
                        fps=15,
                        max_frames=100  # First 100 frames for demo
                    )
                
                break  # Only first sequence per experiment
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Videos saved to: {videos_dir}")


if __name__ == '__main__':
    main()
