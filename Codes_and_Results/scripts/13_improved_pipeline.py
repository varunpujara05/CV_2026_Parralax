"""
Step 13: Improved Pipeline — YOLOv11m + BoT-SORT (No SAHI)
Runs the complete tracking + evaluation pipeline using:
  - Weather-trained YOLOv11m detector
  - BoT-SORT tracker with deep appearance Re-ID
  - Baseline detection only (SAHI toggled OFF)
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
    
    # Use the weather-trained YOLOv11m model
    model_path = project_root / 'runs' / 'detect' / 'visdrone_weather_v11s' / 'weights' / 'best.pt'
    if not model_path.exists():
        print(f"ERROR: Weather-trained model not found at {model_path}")
        print("Please run 12_train_weather_detector.py first!")
        return
    
    dataset_root = project_root / 'VisDrone2019-MOT-train'
    sequences_dir = dataset_root / 'sequences'
    augmented_dir = project_root / 'outputs' / 'augmented'
    tracks_dir = project_root / 'outputs' / 'tracks_improved'
    gt_dir = project_root / 'outputs' / 'gt_mot'
    eval_dir = project_root / 'outputs' / 'eval_results_improved'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Same 5 evaluation sequences as the original pipeline
    all_seqs = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
    eval_seqs = all_seqs[:5]
    
    print("=" * 60)
    print("STEP 13: IMPROVED PIPELINE")
    print("  Detector: YOLOv11m (weather-trained)")
    print("  Tracker: BoT-SORT (deep Re-ID)")
    print("  SAHI: OFF")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Sequences: {eval_seqs}")
    
    # All 10 weather conditions (original + 9 weather combos)
    conditions = [
        ('original', None, None),
        ('rain_light', 'rain', 'light'),
        ('rain_moderate', 'rain', 'moderate'),
        ('rain_severe', 'rain', 'severe'),
        ('fog_light', 'fog', 'light'),
        ('fog_moderate', 'fog', 'moderate'),
        ('fog_severe', 'fog', 'severe'),
        ('dust_light', 'dust', 'light'),
        ('dust_moderate', 'dust', 'moderate'),
        ('dust_severe', 'dust', 'severe'),
    ]
    
    tracker_type = 'botsort'
    
    # --- TRACKING ---
    print("\n" + "=" * 60)
    print("PHASE 1: TRACKING WITH BoT-SORT")
    print("=" * 60)
    
    for condition_name, weather, intensity in conditions:
        experiment_name = f'{tracker_type}_baseline_{condition_name}'
        print(f"\n--- {experiment_name} ---")
        
        for seq_name in eval_seqs:
            track_file = tracks_dir / experiment_name / seq_name / f'{seq_name}.txt'
            if track_file.exists() and track_file.stat().st_size > 0:
                print(f"  Skipping {seq_name} (already tracked)")
                continue
            
            if weather and intensity:
                seq_dir = augmented_dir / f'{weather}_{intensity}' / 'sequences' / seq_name
            else:
                seq_dir = sequences_dir / seq_name
            
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
                use_sahi=False,  # SAHI OFF
                sahi_slice_size=320
            )
            
            save_tracks_mot_format(tracks, str(track_file))
            
            total = sum(len(t) for t in tracks.values())
            print(f"    Total tracked objects: {total}")
    
    # --- EVALUATION ---
    print("\n" + "=" * 60)
    print("PHASE 2: EVALUATION")
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
                    row = overall.iloc[0]
                    print(f"  {exp_name}: MOTA={row.get('MOTA', 0):.1f} | "
                          f"IDF1={row.get('IDF1', 0):.1f} | "
                          f"IDSw={row.get('ID_Switches', 0)}")
        except Exception as e:
            print(f"  Error evaluating {exp_name}: {e}")
    
    # Save combined comparison
    if experiment_results:
        comp_df = compare_experiments(experiment_results, str(eval_dir / 'comparison_improved.csv'))
        print(f"\n--- Improved Pipeline Results ({len(comp_df)} experiments) ---")
        print(comp_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("IMPROVED PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {eval_dir}")
    print(f"Track files saved to: {tracks_dir}")
    print(f"\nNext step: Run 14_compare_results.py to compare old vs new metrics")


if __name__ == '__main__':
    main()
