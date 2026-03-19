"""
Step 10: Run Tracking & Evaluation on Light + Moderate Weather
Extends the pipeline to cover all 9 weather conditions (3 types x 3 intensities).
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

import torch
from src.tracking import run_tracking_on_sequence, save_tracks_mot_format
from src.evaluation import evaluate_experiment, compare_experiments
from pathlib import Path
import pandas as pd


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_path = project_root / 'runs' / 'detect' / 'visdrone_train' / 'weights' / 'best.pt'
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    tracks_dir = project_root / 'outputs' / 'tracks'
    gt_dir = project_root / 'outputs' / 'gt_mot'
    eval_dir = project_root / 'outputs' / 'eval_results'
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Select sequences (same 5 as before)
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    eval_seqs = all_seqs[:5]
    
    print("=" * 60)
    print("STEP 10: LIGHT + MODERATE WEATHER TRACKING & EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Sequences: {eval_seqs}")
    
    # New conditions to process (light + moderate for all weather types)
    new_conditions = [
        ('rain_light', 'rain', 'light'),
        ('rain_moderate', 'rain', 'moderate'),
        ('fog_light', 'fog', 'light'),
        ('fog_moderate', 'fog', 'moderate'),
        ('dust_light', 'dust', 'light'),
        ('dust_moderate', 'dust', 'moderate'),
    ]
    
    tracker_types = ['bytetrack']
    sahi_modes = [False, True]
    
    # --- TRACKING ---
    for tracker_type in tracker_types:
        for use_sahi in sahi_modes:
            sahi_tag = 'sahi' if use_sahi else 'baseline'
            
            for condition_name, weather, intensity in new_conditions:
                experiment_name = f'{tracker_type}_{sahi_tag}_{condition_name}'
                
                print(f"\n--- {experiment_name} ---")
                
                from boxmot import ByteTrack
                
                for seq_name in eval_seqs:
                    track_file = tracks_dir / experiment_name / seq_name / f'{seq_name}.txt'
                    if track_file.exists() and track_file.stat().st_size > 0:
                        print(f"  Skipping {seq_name} (already tracked)")
                        continue
                        
                    seq_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
                    
                    if not seq_dir.exists():
                        print(f"  Skipping {seq_name} (not found)")
                        continue
                    
                    print(f"  Processing: {seq_name}")
                    
                    tracks = run_tracking_on_sequence(
                        model_path=str(model_path),
                        seq_dir=str(seq_dir),
                        tracker_type=tracker_type,
                        conf_thresh=0.25,
                        imgsz=640,
                        device=device,
                        use_sahi=use_sahi,
                        sahi_slice_size=320
                    )
                    
                    track_file = tracks_dir / experiment_name / seq_name / f'{seq_name}.txt'
                    save_tracks_mot_format(tracks, str(track_file))
                    
                    total = sum(len(t) for t in tracks.values())
                    print(f"    Total tracked objects: {total}")
    
    # --- EVALUATION ---
    print("\n" + "=" * 60)
    print("EVALUATING ALL EXPERIMENTS (including new ones)")
    print("=" * 60)
    
    experiment_results = {}
    
    for exp_dir in sorted(tracks_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        result_file = eval_dir / f'{exp_name}_results.csv'
        
        try:
            df = evaluate_experiment(str(gt_dir), str(exp_dir))
            
            if df is not None and not df.empty:
                df.to_csv(result_file, index=False)
                experiment_results[exp_name] = df
                overall = df[df['Sequence'] == 'OVERALL']
                if not overall.empty:
                    mota = overall.iloc[0].get('MOTA', 0)
                    print(f"  {exp_name}: MOTA={mota:.1f}")
        except Exception as e:
            print(f"  Error evaluating {exp_name}: {e}")
    
    # Save combined comparison
    if experiment_results:
        comp_df = compare_experiments(experiment_results, str(eval_dir / 'comparison.csv'))
        print(f"\nSaved comparison with {len(comp_df)} experiments")
        print("\n" + comp_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
