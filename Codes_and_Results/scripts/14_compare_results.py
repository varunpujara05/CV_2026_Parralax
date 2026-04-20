"""
Step 14: Compare Old vs Improved Pipeline Results
Generates side-by-side comparison tables and plots showing metric improvements.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd


def load_comparison_csv(csv_path):
    """Load a comparison CSV and return as DataFrame."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    old_eval_dir = project_root / 'outputs' / 'eval_results'
    new_eval_dir = project_root / 'outputs' / 'eval_results_improved'
    output_dir = project_root / 'outputs' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 14: COMPARE OLD vs IMPROVED PIPELINE")
    print("=" * 60)
    
    # Load old results (baseline only — ignore SAHI runs)
    old_results = {}
    if old_eval_dir.exists():
        for csv_file in sorted(old_eval_dir.glob('bytetrack_baseline_*_results.csv')):
            exp_name = csv_file.stem.replace('_results', '')
            df = pd.read_csv(csv_file)
            overall = df[df['Sequence'] == 'OVERALL']
            if not overall.empty:
                old_results[exp_name] = overall.iloc[0]
    
    # Load new results
    new_results = {}
    if new_eval_dir.exists():
        for csv_file in sorted(new_eval_dir.glob('botsort_baseline_*_results.csv')):
            exp_name = csv_file.stem.replace('_results', '')
            df = pd.read_csv(csv_file)
            overall = df[df['Sequence'] == 'OVERALL']
            if not overall.empty:
                new_results[exp_name] = overall.iloc[0]
    
    if not old_results:
        print("WARNING: No old results found in outputs/eval_results/")
    if not new_results:
        print("WARNING: No new results found in outputs/eval_results_improved/")
        print("Please run 13_improved_pipeline.py first!")
        return
    
    # Build comparison table
    comparison_rows = []
    metrics = ['MOTA', 'MOTP', 'IDF1', 'HOTA', 'Precision', 'Recall', 'ID_Switches', 'FN']
    
    # Map old experiment names to weather conditions
    conditions = [
        'original', 'rain_light', 'rain_moderate', 'rain_severe',
        'fog_light', 'fog_moderate', 'fog_severe',
        'dust_light', 'dust_moderate', 'dust_severe'
    ]
    
    for condition in conditions:
        old_key = f'bytetrack_baseline_{condition}'
        new_key = f'botsort_baseline_{condition}'
        
        row = {'Condition': condition}
        
        if old_key in old_results:
            for m in metrics:
                row[f'Old_{m}'] = old_results[old_key].get(m, 0)
        else:
            for m in metrics:
                row[f'Old_{m}'] = 'N/A'
        
        if new_key in new_results:
            for m in metrics:
                row[f'New_{m}'] = new_results[new_key].get(m, 0)
        else:
            for m in metrics:
                row[f'New_{m}'] = 'N/A'
        
        # Calculate deltas for key metrics
        for m in ['MOTA', 'IDF1', 'HOTA', 'Precision', 'Recall']:
            old_val = row.get(f'Old_{m}', 'N/A')
            new_val = row.get(f'New_{m}', 'N/A')
            if old_val != 'N/A' and new_val != 'N/A':
                row[f'Delta_{m}'] = round(float(new_val) - float(old_val), 2)
            else:
                row[f'Delta_{m}'] = 'N/A'
        
        # ID Switches delta (lower is better, so negative delta is good)
        old_ids = row.get('Old_ID_Switches', 'N/A')
        new_ids = row.get('New_ID_Switches', 'N/A')
        if old_ids != 'N/A' and new_ids != 'N/A':
            row['Delta_ID_Switches'] = int(new_ids) - int(old_ids)
        else:
            row['Delta_ID_Switches'] = 'N/A'
        
        comparison_rows.append(row)
    
    comp_df = pd.DataFrame(comparison_rows)
    
    # --- Print Summary Table ---
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON: Old (YOLOv8s + ByteTrack) vs New (YOLOv11m + BoT-SORT)")
    print("=" * 80)
    
    # Print a focused comparison
    summary_cols = ['Condition', 'Old_MOTA', 'New_MOTA', 'Delta_MOTA', 
                    'Old_IDF1', 'New_IDF1', 'Delta_IDF1',
                    'Old_ID_Switches', 'New_ID_Switches', 'Delta_ID_Switches']
    
    available_cols = [c for c in summary_cols if c in comp_df.columns]
    if available_cols:
        print(comp_df[available_cols].to_string(index=False))
    else:
        print(comp_df.to_string(index=False))
    
    # Save full comparison
    comp_df.to_csv(output_dir / 'old_vs_new_comparison.csv', index=False)
    
    # --- Calculate Aggregate Improvements ---
    print("\n" + "=" * 60)
    print("AGGREGATE IMPROVEMENTS")
    print("=" * 60)
    
    for m in ['MOTA', 'IDF1', 'HOTA', 'Precision', 'Recall']:
        delta_col = f'Delta_{m}'
        if delta_col in comp_df.columns:
            deltas = comp_df[delta_col]
            numeric_deltas = pd.to_numeric(deltas, errors='coerce').dropna()
            if len(numeric_deltas) > 0:
                avg_delta = numeric_deltas.mean()
                sign = "+" if avg_delta > 0 else ""
                print(f"  {m}: {sign}{avg_delta:.2f}% average improvement")
    
    delta_ids = comp_df.get('Delta_ID_Switches', pd.Series())
    numeric_ids = pd.to_numeric(delta_ids, errors='coerce').dropna()
    if len(numeric_ids) > 0:
        avg_ids = numeric_ids.mean()
        sign = "+" if avg_ids > 0 else ""
        print(f"  ID Switches: {sign}{avg_ids:.0f} average change (negative = better)")
    
    # --- Generate Comparison Report ---
    report_path = output_dir / 'improvement_report.md'
    with open(report_path, 'w') as f:
        f.write("# Pipeline Improvement Report\n\n")
        f.write("## Old Pipeline: YOLOv8s (clean-trained) + ByteTrack + SAHI\n")
        f.write("## New Pipeline: YOLOv11m (weather-trained) + BoT-SORT (No SAHI)\n\n")
        f.write("### Side-by-Side Comparison\n\n")
        f.write(comp_df[available_cols].to_markdown(index=False) if available_cols 
                else comp_df.to_markdown(index=False))
        f.write("\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Comparison CSV: {output_dir / 'old_vs_new_comparison.csv'}")
    print(f"Report: {report_path}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
