"""
Evaluation Module using MOT metrics.
Computes MOTA, HOTA, IDF1, ID switches, false negatives.
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_mot_file(filepath):
    """
    Load a MOT format file.
    
    Format: frame,id,x,y,w,h,conf,class,visibility
    
    Returns:
        dict: {frame_id: [(id, x, y, w, h, conf), ...]}
    """
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(row[0])
            tid = int(row[1])
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            conf = float(row[6]) if len(row) > 6 else 1.0
            data[frame].append({
                'id': tid, 'bbox': (x, y, w, h), 'conf': conf
            })
    return data


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x,y,w,h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)
    
    if xb <= xa or yb <= ya:
        return 0.0
    
    intersection = (xb - xa) * (yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_mot_metrics(gt_data, pred_data, iou_threshold=0.5):
    """
    Compute MOT metrics for a single sequence.
    
    Metrics:
        - MOTA (Multiple Object Tracking Accuracy)
        - IDF1 (ID F1 Score)
        - ID Switches
        - False Positives (FP)
        - False Negatives (FN / Misses)
        - True Positives
        - Precision, Recall
        - MOTP (Multiple Object Tracking Precision)
        
    Args:
        gt_data: Ground truth {frame: [{id, bbox, conf}, ...]}
        pred_data: Predictions {frame: [{id, bbox, conf}, ...]}
        iou_threshold: IoU threshold for matching
        
    Returns:
        dict with computed metrics
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_id_switches = 0
    total_iou_sum = 0
    
    # Track ID assignment mapping (gt_id -> last matched pred_id)
    id_mapping = {}
    
    # For IDF1 (identity metrics)
    gt_id_frames = defaultdict(set)    # gt_id -> set of frames it appears
    pred_id_frames = defaultdict(set)  # pred_id -> set of frames it appears
    match_frames = defaultdict(lambda: defaultdict(set))  # gt_id -> pred_id -> frames matched
    
    for frame_id in all_frames:
        gt_objs = gt_data.get(frame_id, [])
        pred_objs = pred_data.get(frame_id, [])
        
        n_gt = len(gt_objs)
        n_pred = len(pred_objs)
        total_gt += n_gt
        
        # Track GT IDs appearance
        for gt_obj in gt_objs:
            gt_id_frames[gt_obj['id']].add(frame_id)
        for pred_obj in pred_objs:
            pred_id_frames[pred_obj['id']].add(frame_id)
        
        if n_gt == 0 and n_pred == 0:
            continue
        
        if n_gt == 0:
            total_fp += n_pred
            continue
        
        if n_pred == 0:
            total_fn += n_gt
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((n_gt, n_pred))
        for gi, gt_obj in enumerate(gt_objs):
            for pi, pred_obj in enumerate(pred_objs):
                iou_matrix[gi, pi] = compute_iou(gt_obj['bbox'], pred_obj['bbox'])
        
        # Greedy matching (by highest IoU)
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        # Sort all pairs by IoU descending
        pairs = []
        for gi in range(n_gt):
            for pi in range(n_pred):
                if iou_matrix[gi, pi] >= iou_threshold:
                    pairs.append((gi, pi, iou_matrix[gi, pi]))
        
        pairs.sort(key=lambda x: -x[2])
        
        for gi, pi, iou_val in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append((gi, pi, iou_val))
        
        tp = len(matches)
        fp = n_pred - tp
        fn = n_gt - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Check for ID switches
        for gi, pi, iou_val in matches:
            gt_id = gt_objs[gi]['id']
            pred_id = pred_objs[pi]['id']
            total_iou_sum += iou_val
            
            # Record match for IDF1
            match_frames[gt_id][pred_id].add(frame_id)
            
            # Check ID switch
            if gt_id in id_mapping:
                if id_mapping[gt_id] != pred_id:
                    total_id_switches += 1
            id_mapping[gt_id] = pred_id
    
    # Compute MOTA
    if total_gt > 0:
        mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt
    else:
        mota = 0.0
    
    # Compute MOTP (average IoU of matched pairs)
    motp = total_iou_sum / total_tp if total_tp > 0 else 0.0
    
    # Compute precision & recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # Compute IDF1
    # Find best GT-Pred ID mapping
    idf1 = compute_idf1(gt_id_frames, pred_id_frames, match_frames)
    
    # Compute simplified HOTA
    hota = compute_simple_hota(gt_data, pred_data, iou_threshold)
    
    return {
        'MOTA': round(mota * 100, 2),
        'MOTP': round(motp * 100, 2),
        'IDF1': round(idf1 * 100, 2),
        'HOTA': round(hota * 100, 2),
        'Precision': round(precision * 100, 2),
        'Recall': round(recall * 100, 2),
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'ID_Switches': total_id_switches,
        'Total_GT': total_gt,
    }


def compute_idf1(gt_id_frames, pred_id_frames, match_frames):
    """
    Compute IDF1 score using greedy ID matching.
    
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """
    # Build cost matrix for GT-Pred ID assignment
    gt_ids = list(gt_id_frames.keys())
    pred_ids = list(pred_id_frames.keys())
    
    if not gt_ids or not pred_ids:
        return 0.0
    
    # Greedy match: pick GT-Pred pair with most co-occurring frames
    best_matches = {}
    matched_pred = set()
    
    assignment_scores = []
    for gid in gt_ids:
        for pid in pred_ids:
            n_matched = len(match_frames[gid].get(pid, set()))
            if n_matched > 0:
                assignment_scores.append((gid, pid, n_matched))
    
    assignment_scores.sort(key=lambda x: -x[2])
    
    matched_gt_ids = set()
    for gid, pid, score in assignment_scores:
        if gid in matched_gt_ids or pid in matched_pred:
            continue
        best_matches[gid] = pid
        matched_gt_ids.add(gid)
        matched_pred.add(pid)
    
    # Compute IDTP, IDFP, IDFN
    idtp = 0
    for gid, pid in best_matches.items():
        idtp += len(match_frames[gid][pid])
    
    total_gt_frames = sum(len(frames) for frames in gt_id_frames.values())
    total_pred_frames = sum(len(frames) for frames in pred_id_frames.values())
    
    idfn = total_gt_frames - idtp
    idfp = total_pred_frames - idtp
    
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
    
    return idf1


def compute_simple_hota(gt_data, pred_data, iou_threshold=0.5):
    """
    Compute a simplified HOTA (Higher Order Tracking Accuracy).
    
    HOTA = sqrt(DetA * AssA) where:
    - DetA = TP / (TP + FP + FN) (detection accuracy)
    - AssA = association accuracy (correct ID assignments)
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    tp = 0
    fp = 0
    fn = 0
    correct_assoc = 0
    total_assoc = 0
    
    id_mapping = {}
    
    for frame_id in all_frames:
        gt_objs = gt_data.get(frame_id, [])
        pred_objs = pred_data.get(frame_id, [])
        
        n_gt = len(gt_objs)
        n_pred = len(pred_objs)
        
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        
        # IoU matching
        iou_matrix = np.zeros((n_gt, n_pred))
        for gi, gt_obj in enumerate(gt_objs):
            for pi, pred_obj in enumerate(pred_objs):
                iou_matrix[gi, pi] = compute_iou(gt_obj['bbox'], pred_obj['bbox'])
        
        matched_gt = set()
        matched_pred = set()
        
        pairs = []
        for gi in range(n_gt):
            for pi in range(n_pred):
                if iou_matrix[gi, pi] >= iou_threshold:
                    pairs.append((gi, pi, iou_matrix[gi, pi]))
        pairs.sort(key=lambda x: -x[2])
        
        for gi, pi, iou_val in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            
            gt_id = gt_objs[gi]['id']
            pred_id = pred_objs[pi]['id']
            
            tp += 1
            total_assoc += 1
            
            if gt_id in id_mapping and id_mapping[gt_id] == pred_id:
                correct_assoc += 1
            elif gt_id not in id_mapping:
                correct_assoc += 1
                id_mapping[gt_id] = pred_id
        
        fp += n_pred - len(matched_pred)
        fn += n_gt - len(matched_gt)
    
    det_a = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    ass_a = correct_assoc / total_assoc if total_assoc > 0 else 0.0
    hota = np.sqrt(det_a * ass_a)
    
    return hota


def evaluate_experiment(gt_dir, pred_dir, sequence_names=None):
    """
    Evaluate tracking results for an experiment.
    
    Args:
        gt_dir: Directory with GT files (seq_name/gt/gt.txt)
        pred_dir: Directory with prediction files (seq_name/seq_name.txt)
        sequence_names: Optional list of sequences to evaluate
        
    Returns:
        DataFrame with per-sequence and aggregate metrics
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    if sequence_names is None:
        sequence_names = sorted([d.name for d in pred_dir.iterdir() if d.is_dir()])
    
    results = []
    
    for seq_name in tqdm(sequence_names, desc="Evaluating"):
        gt_file = gt_dir / seq_name / 'gt' / 'gt.txt'
        pred_file = pred_dir / seq_name / f'{seq_name}.txt'
        
        if not gt_file.exists():
            print(f"GT not found: {gt_file}")
            continue
        if not pred_file.exists():
            print(f"Prediction not found: {pred_file}")
            continue
        
        gt_data = load_mot_file(gt_file)
        pred_data = load_mot_file(pred_file)
        
        metrics = compute_mot_metrics(gt_data, pred_data)
        metrics['Sequence'] = seq_name
        results.append(metrics)
    
    if not results:
        print("No sequences evaluated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Compute aggregate metrics
    agg = {
        'Sequence': 'OVERALL',
        'MOTA': df['MOTA'].mean(),
        'MOTP': df['MOTP'].mean(),
        'IDF1': df['IDF1'].mean(),
        'HOTA': df['HOTA'].mean(),
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'TP': df['TP'].sum(),
        'FP': df['FP'].sum(),
        'FN': df['FN'].sum(),
        'ID_Switches': df['ID_Switches'].sum(),
        'Total_GT': df['Total_GT'].sum(),
    }
    
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    
    return df


def compare_experiments(experiment_results, save_path=None):
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_results: dict of {experiment_name: DataFrame}
        save_path: Optional path to save comparison CSV
        
    Returns:
        DataFrame with comparison
    """
    comparison = []
    
    for name, df in experiment_results.items():
        overall = df[df['Sequence'] == 'OVERALL'].iloc[0]
        comparison.append({
            'Experiment': name,
            'MOTA': overall['MOTA'],
            'MOTP': overall['MOTP'],
            'IDF1': overall['IDF1'],
            'HOTA': overall['HOTA'],
            'Precision': overall['Precision'],
            'Recall': overall['Recall'],
            'ID_Switches': overall['ID_Switches'],
            'FN': overall['FN'],
        })
    
    comp_df = pd.DataFrame(comparison)
    
    if save_path:
        comp_df.to_csv(save_path, index=False)
        print(f"Comparison saved to {save_path}")
    
    return comp_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MOT Evaluation')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='Ground truth directory')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Predictions directory')
    parser.add_argument('--output', type=str, default='outputs/eval_results/results.csv')
    
    args = parser.parse_args()
    
    df = evaluate_experiment(args.gt_dir, args.pred_dir)
    
    if not df.empty:
        print("\n=== Evaluation Results ===")
        print(df.to_string(index=False))
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
