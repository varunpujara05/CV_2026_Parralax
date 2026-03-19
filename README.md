# UAV-Based Multi-Object Tracking Pipeline

**Robust MOT under Synthetic Adverse Weather Conditions**

A research-grade multi-object tracking system for UAV (drone) imagery using the VisDrone2019-MOT dataset. The pipeline handles detection of small, dense objects and maintains tracking consistency under synthetically generated weather degradation (rain, fog, dust storms).

## Architecture

```
Detection (YOLOv8 + SAHI) → Tracking (ByteTrack / DeepSORT) → Evaluation (MOTA/HOTA/IDF1)
        ↑                                                              ↓
  Weather Engine ──────────────────────────────────────────→ Analytics & Visualizations
  (Rain/Fog/Dust)
```

## Project Structure

```
├── src/                          # Core modules
│   ├── dataset_utils.py          # VisDrone parsing & YOLO conversion
│   ├── weather_engine.py         # Synthetic weather augmentation
│   ├── detection.py              # YOLOv8 + SAHI detection pipeline
│   ├── tracking.py               # BoxMOT tracking (ByteTrack/DeepSORT)
│   ├── evaluation.py             # MOT metrics computation
│   ├── visualization.py          # Plots, videos, comparison grids
│   └── analysis.py               # Advanced analysis & report generation
├── scripts/                      # Pipeline execution scripts
│   ├── 01_prepare_dataset.py     # Parse & convert VisDrone dataset
│   ├── 02_generate_weather.py    # Apply weather augmentations
│   ├── 03_train_detector.py      # Fine-tune YOLOv8
│   ├── 04_run_detection.py       # Run detection (baseline + SAHI)
│   ├── 05_run_tracking.py        # Run tracking pipelines
│   ├── 06_evaluate.py            # Compute MOT metrics
│   ├── 07_visualize.py           # Generate all visualizations
│   └── 08_analysis.py            # Advanced analysis & report
├── config/visdrone.yaml          # YOLO dataset configuration
├── VisDrone2019-MOT-train/       # Dataset
├── outputs/                      # All generated outputs
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline (Step by Step)
```bash
# Step 1: Prepare dataset (VisDrone → YOLO format)
python scripts/01_prepare_dataset.py

# Step 2: Generate weather augmentations
python scripts/02_generate_weather.py

# Step 3: Train YOLOv8 detector
python scripts/03_train_detector.py

# Step 4: Run detection (baseline + SAHI)
python scripts/04_run_detection.py

# Step 5: Run multi-object tracking
python scripts/05_run_tracking.py

# Step 6: Evaluate tracking metrics
python scripts/06_evaluate.py

# Step 7: Generate visualizations
python scripts/07_visualize.py

# Step 8: Advanced analysis & report
python scripts/08_analysis.py
```

## Key Features

- **Synthetic Weather Engine**: Rain, fog, dust storms at 3 intensity levels (light/moderate/severe)
- **Small Object Detection**: YOLOv8 + SAHI tiling for small object enhancement
- **Multi-Object Tracking**: ByteTrack & DeepSORT via BoxMOT
- **Comprehensive Evaluation**: MOTA, HOTA, IDF1, ID switches, false negatives
- **Rich Visualizations**: Weather comparisons, tracking videos, performance dashboards
- **Advanced Analysis**: Localization drift, failure cases, ablation studies

## Metrics Computed

| Metric | Description |
|--------|-------------|
| MOTA | Multiple Object Tracking Accuracy |
| HOTA | Higher Order Tracking Accuracy |
| IDF1 | ID F1 Score (identity preservation) |
| MOTP | Multiple Object Tracking Precision |
| ID Switches | Number of identity changes |
| FN | False Negatives (missed detections) |

## Dataset

**VisDrone2019-MOT** — UAV-based multi-object tracking benchmark
- 56 training sequences
- Objects: pedestrians, vehicles, bicycles, etc.
- Challenges: small object sizes, high density, occlusion

## Requirements

- Python 3.8+
- CUDA GPU recommended (for training/inference)
- ~10GB disk space (dataset + augmented data)
