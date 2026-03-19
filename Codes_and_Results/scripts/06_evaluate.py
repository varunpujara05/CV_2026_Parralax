"""
Step 6: Evaluation
Compute MOT metrics (MOTA, HOTA, IDF1, ID switches, FN) for all experiments.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_experiment, compare_experiments
from pathlib import Path
import pandas as pd


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    gt_dir = project_root / 'outputs' / 'gt_mot'
    tracks_dir = project_root / 'outputs' / 'tracks'
    eval_dir = project_root / 'outputs' / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 6: EVALUATION")
    print("=" * 60)
    
    if not gt_dir.exists():
        print("ERROR: Ground truth not found. Run 01_prepare_dataset.py first!")
        return
    
    # Find all experiment directories
    experiments = {}
    
    if tracks_dir.exists():
        for exp_dir in sorted(tracks_dir.iterdir()):
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                print(f"\n--- Evaluating: {exp_name} ---")
                
                df = evaluate_experiment(str(gt_dir), str(exp_dir))
                
                if not df.empty:
                    experiments[exp_name] = df
                    
                    # Save per-experiment results
                    df.to_csv(eval_dir / f'{exp_name}_results.csv', index=False)
                    
                    # Print summary
                    overall = df[df['Sequence'] == 'OVERALL']
                    if not overall.empty:
                        row = overall.iloc[0]
                        print(f"  MOTA: {row['MOTA']:.1f}%")
                        print(f"  HOTA: {row['HOTA']:.1f}%")
                        print(f"  IDF1: {row['IDF1']:.1f}%")
                        print(f"  ID Switches: {row['ID_Switches']}")
                        print(f"  FN: {row['FN']}")
    
    # Compare all experiments
    if experiments:
        print("\n--- Experiment Comparison ---")
        comparison = compare_experiments(
            experiments,
            save_path=str(eval_dir / 'comparison.csv')
        )
        print(comparison.to_string(index=False))
    else:
        print("\nNo tracking experiments found. Run 05_run_tracking.py first!")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {eval_dir}")


if __name__ == '__main__':
    main()
