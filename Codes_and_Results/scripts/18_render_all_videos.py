"""
Step 18: Render Full Videos
Renders enhanced tracking videos for ALL 5 evaluation sequences, with ALL frames,
for BOTH ByteTrack and BoT-SORT across all weather conditions.
Warning: This might take some time and consume significant disk space.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
enhanced_videos = importlib.import_module('scripts.09_enhanced_videos')
render_enhanced_video = enhanced_videos.render_enhanced_video


def generate_videos_for_tracker(tracks_dir, videos_dir, tracker_name, project_root):
    """Generate videos for all 5 sequences and all frames."""
    print(f"\n  Generating Full Videos for {tracker_name.upper()}...")
    
    sequences_dir = project_root / 'VisDrone2019-MOT-train' / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    
    # All 5 evaluation sequences
    eval_seqs = [
        'uav0000013_00000_v', 
        'uav0000013_01073_v', 
        'uav0000013_01392_v', 
        'uav0000020_00406_v', 
        'uav0000071_03240_v'
    ]
    
    for exp_dir in sorted(tracks_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        print(f"\n    Rendering [{exp_name}]")
        
        for seq_name in eval_seqs:
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
                print(f"      Missing source frames: {frames_dir}")
                continue
                
            video_path = videos_dir / f'{exp_name}_{seq_name}.mp4'
            
            # If the video exists and has a reasonable file size (> 1MB), assume it's done and skip to resume
            if video_path.exists():
                if video_path.stat().st_size > 1024 * 1024:
                    print(f"      Skipping {video_path.name} (Already rendered)")
                    continue
                else:
                    print(f"      Overwriting incomplete video {video_path.name}")
                    video_path.unlink()
                
            render_enhanced_video(
                str(frames_dir),
                str(track_file),
                str(video_path),
                condition_label=exp_name,
                fps=15,
                max_frames=None  # Remove the limit, render ALL frames!
            )


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_tracks_dir = project_root / 'outputs' / 'tracks_improved'
    
    trackers = ['bytetrack', 'botsort']
    
    print("=" * 60)
    print("STEP 18: RENDER FULL EVALUATION VIDEOS")
    print("=" * 60)
    
    for tracker in trackers:
        # We already separated tracks into tracks_bytetrack and tracks_botsort in step 17
        tracks_out = project_root / 'outputs' / f'tracks_{tracker}'
        videos_out = project_root / 'outputs' / f'videos_{tracker}'
        
        videos_out.mkdir(parents=True, exist_ok=True)
        
        if not tracks_out.exists():
            print(f"WARNING: Track directory missing for {tracker}. Looking in {tracks_out}")
            continue
            
        generate_videos_for_tracker(tracks_out, videos_out, tracker, project_root)
        
    print("\n" + "=" * 60)
    print("FULL VIDEO RENDERING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
