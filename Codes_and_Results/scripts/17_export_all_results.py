"""
Step 17: Export & Separate Results, Plots, and Videos for Both Trackers
Organizes and generates all outputs (results, videos, plots) into specific
folders suffixed by the tracker name (_bytetrack and _botsort).
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import importlib
enhanced_videos = importlib.import_module('scripts.09_enhanced_videos')
render_enhanced_video = enhanced_videos.render_enhanced_video


def generate_plots_for_tracker(eval_dir, output_dir, tracker_name):
    """Generate plots specifically for a given tracker's evaluation directory."""
    print(f"  Generating plots for {tracker_name}...")
    
    # 1. Weather Impact Plot
    csv_path = eval_dir / f'comparison_improved_{tracker_name}.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Filter only baseline (no SAHI) since we turned it off
        df_filtered = df[df['Experiment'].str.contains('baseline')]
        
        if not df_filtered.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_filtered, x='Experiment', y='MOTA', color='skyblue')
            plt.title(f'MOTA vs Weather Conditions ({tracker_name})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'mota_vs_weather.png', dpi=300)
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_filtered, x='Experiment', y='ID_Switches', color='salmon')
            plt.title(f'ID Switches vs Weather Conditions ({tracker_name})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'idsw_vs_weather.png', dpi=300)
            plt.close()


def generate_videos_for_tracker(tracks_dir, videos_dir, tracker_name, project_root):
    """Generate videos specifically for a given tracker."""
    print(f"  Generating videos for {tracker_name}...")
    
    sequences_dir = project_root / 'VisDrone2019-MOT-train' / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    
    # Select 2 representative sequences to speed up generation
    test_seqs = ['uav0000013_00000_v', 'uav0000020_00406_v']
    
    for exp_dir in sorted(tracks_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        
        for seq_name in test_seqs:
            seq_dir_track = exp_dir / seq_name
            track_file = seq_dir_track / f'{seq_name}.txt'
            if not track_file.exists():
                continue
            
            # Find frame directory
            if 'original' in exp_name:
                frames_dir = sequences_dir / seq_name
            else:
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
                continue
                
            video_path = videos_dir / f'{exp_name}_{seq_name}.mp4'
            if not video_path.exists():
                render_enhanced_video(
                    str(frames_dir),
                    str(track_file),
                    str(video_path),
                    condition_label=exp_name,
                    fps=15,
                    max_frames=100  # Only 100 frames to save time
                )


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_eval_dir = project_root / 'outputs' / 'eval_results_improved'
    source_tracks_dir = project_root / 'outputs' / 'tracks_improved'
    
    trackers = ['bytetrack', 'botsort']
    
    print("=" * 60)
    print("STEP 17: EXPORT & SEPARATE TRACKER OUTPUTS")
    print("=" * 60)
    
    for tracker in trackers:
        print(f"\nProcessing {tracker.upper()}...")
        
        # Define tracker-specific output dirs
        eval_out = project_root / 'outputs' / f'eval_results_{tracker}'
        tracks_out = project_root / 'outputs' / f'tracks_{tracker}'
        plots_out = project_root / 'outputs' / f'plots_{tracker}'
        videos_out = project_root / 'outputs' / f'videos_{tracker}'
        
        eval_out.mkdir(parents=True, exist_ok=True)
        tracks_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)
        videos_out.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy Results (CSVs)
        print(f"  Copying results to {eval_out.name}...")
        results_dfs = []
        for csv_file in source_eval_dir.glob(f'{tracker}_baseline_*_results.csv'):
            shutil.copy2(csv_file, eval_out / csv_file.name)
            
            # Read to build combined comparison for this tracker
            df = pd.read_csv(csv_file)
            overall = df[df['Sequence'] == 'OVERALL']
            if not overall.empty:
                overall = overall.copy()
                overall['Experiment'] = csv_file.stem.replace('_results', '')
                results_dfs.append(overall)
                
        if results_dfs:
            combined_df = pd.concat(results_dfs, ignore_index=True)
            combined_df.to_csv(eval_out / f'comparison_improved_{tracker}.csv', index=False)
            
        # 2. Copy Tracks
        print(f"  Copying tracks to {tracks_out.name}...")
        for exp_dir in source_tracks_dir.glob(f'{tracker}_baseline_*'):
            if exp_dir.is_dir():
                dest_dir = tracks_out / exp_dir.name
                if not dest_dir.exists():
                    shutil.copytree(exp_dir, dest_dir)
                    
        # 3. Generate Plots
        generate_plots_for_tracker(eval_out, plots_out, tracker)
        
        # 4. Generate Videos
        generate_videos_for_tracker(tracks_out, videos_out, tracker, project_root)
        
    print("\n" + "=" * 60)
    print("EXPORT & SEPARATION COMPLETE!")
    print("=" * 60)
    print("Everything has been uniquely saved into:")
    print("  - outputs/eval_results_bytetrack")
    print("  - outputs/plots_bytetrack")
    print("  - outputs/videos_bytetrack")
    print("  - outputs/eval_results_botsort")
    print("  - outputs/plots_botsort")
    print("  - outputs/videos_botsort")


if __name__ == '__main__':
    main()
