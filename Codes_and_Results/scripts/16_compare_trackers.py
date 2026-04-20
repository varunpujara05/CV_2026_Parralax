"""
Step 16: Compare Trackers (BoT-SORT vs ByteTrack)
Generates a side-by-side comparison showing the difference between BoT-SORT
and ByteTrack using the identical YOLOv11s weather-trained detector.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd


def main():
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    eval_dir = project_root / 'outputs' / 'eval_results_improved'
    output_dir = project_root / 'outputs' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 16: COMPARE BoT-SORT vs ByteTrack (on YOLOv11s)")
    print("=" * 60)
    
    # Load ByteTrack results
    bytetrack_results = {}
    if eval_dir.exists():
        for csv_file in sorted(eval_dir.glob('bytetrack_baseline_*_results.csv')):
            exp_name = csv_file.stem.replace('_results', '')
            df = pd.read_csv(csv_file)
            overall = df[df['Sequence'] == 'OVERALL']
            if not overall.empty:
                bytetrack_results[exp_name] = overall.iloc[0]
    
    # Load BoT-SORT results
    botsort_results = {}
    if eval_dir.exists():
        for csv_file in sorted(eval_dir.glob('botsort_baseline_*_results.csv')):
            exp_name = csv_file.stem.replace('_results', '')
            df = pd.read_csv(csv_file)
            overall = df[df['Sequence'] == 'OVERALL']
            if not overall.empty:
                botsort_results[exp_name] = overall.iloc[0]
    
    if not bytetrack_results:
        print("WARNING: No ByteTrack results found. Run 15_ablation_bytetrack.py first!")
        return
    
    # Build comparison table
    comparison_rows = []
    metrics = ['MOTA', 'MOTP', 'IDF1', 'HOTA', 'Precision', 'Recall', 'ID_Switches', 'FN']
    
    conditions = [
        'original', 'rain_light', 'rain_moderate', 'rain_severe',
        'fog_light', 'fog_moderate', 'fog_severe',
        'dust_light', 'dust_moderate', 'dust_severe'
    ]
    
    for condition in conditions:
        byte_key = f'bytetrack_baseline_{condition}'
        bot_key = f'botsort_baseline_{condition}'
        
        row = {'Condition': condition}
        
        if byte_key in bytetrack_results:
            for m in metrics:
                row[f'Byte_{m}'] = bytetrack_results[byte_key].get(m, 0)
        else:
            for m in metrics:
                row[f'Byte_{m}'] = 'N/A'
        
        if bot_key in botsort_results:
            for m in metrics:
                row[f'BoT_{m}'] = botsort_results[bot_key].get(m, 0)
        else:
            for m in metrics:
                row[f'BoT_{m}'] = 'N/A'
        
        # Calculate deltas (BoT-SORT - ByteTrack)
        for m in ['MOTA', 'IDF1', 'HOTA', 'Precision', 'Recall']:
            byte_val = row.get(f'Byte_{m}', 'N/A')
            bot_val = row.get(f'BoT_{m}', 'N/A')
            if byte_val != 'N/A' and bot_val != 'N/A':
                # Positive delta means BoT-SORT is better
                row[f'Delta_{m}'] = round(float(bot_val) - float(byte_val), 2)
            else:
                row[f'Delta_{m}'] = 'N/A'
        
        # ID Switches delta (lower is better, so negative is good)
        byte_ids = row.get('Byte_ID_Switches', 'N/A')
        bot_ids = row.get('BoT_ID_Switches', 'N/A')
        if byte_ids != 'N/A' and bot_ids != 'N/A':
            row['Delta_ID_Switches'] = int(bot_ids) - int(byte_ids)
        else:
            row['Delta_ID_Switches'] = 'N/A'
        
        comparison_rows.append(row)
    
    comp_df = pd.DataFrame(comparison_rows)
    
    # --- Print Summary Table ---
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON: ByteTrack vs BoT-SORT (both using YOLOv11s)")
    print("=" * 80)
    
    summary_cols = ['Condition', 'Byte_MOTA', 'BoT_MOTA', 'Delta_MOTA', 
                    'Byte_IDF1', 'BoT_IDF1', 'Delta_IDF1',
                    'Byte_ID_Switches', 'BoT_ID_Switches', 'Delta_ID_Switches']
    
    available_cols = [c for c in summary_cols if c in comp_df.columns]
    if available_cols:
        print(comp_df[available_cols].to_string(index=False))
    
    # Save full comparison
    comp_df.to_csv(output_dir / 'bytetrack_vs_botsort_comparison.csv', index=False)
    
    # --- Calculate Aggregate Improvements ---
    print("\n" + "=" * 60)
    print("ADVANTAGE OF BoT-SORT OVER BYTETRACK")
    print("=" * 60)
    
    for m in ['MOTA', 'IDF1', 'HOTA', 'Precision', 'Recall']:
        delta_col = f'Delta_{m}'
        if delta_col in comp_df.columns:
            deltas = comp_df[delta_col]
            numeric_deltas = pd.to_numeric(deltas, errors='coerce').dropna()
            if len(numeric_deltas) > 0:
                avg_delta = numeric_deltas.mean()
                sign = "+" if avg_delta > 0 else ""
                print(f"  {m}: {sign}{avg_delta:.2f}% avg diff")
    
    delta_ids = comp_df.get('Delta_ID_Switches', pd.Series())
    numeric_ids = pd.to_numeric(delta_ids, errors='coerce').dropna()
    if len(numeric_ids) > 0:
        sum_ids = numeric_ids.sum()
        sign = "+" if sum_ids > 0 else ""
        print(f"  ID Switches: {sign}{sum_ids:.0f} total diff across all conds (negative = BoT-SORT better)")
    
    print(f"\nSaved CSV: {output_dir / 'bytetrack_vs_botsort_comparison.csv'}")

if __name__ == '__main__':
    main()
