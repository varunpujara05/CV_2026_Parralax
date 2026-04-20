# UAV Multi-Object Tracking Under Adverse Weather

This repository contains an end-to-end computer vision pipeline for UAV multi-object tracking (MOT) on VisDrone2019-MOT, with synthetic weather degradation and comparative tracking experiments.

The project supports:
- dataset conversion and preparation
- weather generation (rain, fog, dust at light/moderate/severe levels)
- detector training and inference
- tracking with ByteTrack, BoT-SORT, and optional DeepSORT support in code
- MOT evaluation and experiment comparison
- report-ready plots, tables, and rendered videos

## Pipeline Summary

```
Dataset Prep -> Weather Augmentation -> Detection -> Tracking -> Evaluation -> Analysis/Visualization
```

Core flow:
- detector: YOLO models via Ultralytics, with optional SAHI slicing for small-object scenarios
- tracker backend: BoxMOT wrappers in src/tracking.py
- metrics: standard MOT metrics (MOTA, MOTP, IDF1, HOTA, Precision, Recall, ID switches, FN)

## Repository Layout

```
config/                     # Dataset and weather configuration YAML files
data/                       # YOLO-format datasets (base + weather variants)
outputs/                    # Tracking files, eval CSVs, plots, videos, reports
runs/                       # Training and detection run artifacts
scripts/                    # Numbered runnable pipeline stages (01-18)
src/                        # Core implementation modules
VisDrone2019-MOT-train/     # Source dataset (sequences + annotations)
requirements.txt            # Python dependencies
README.md
```

## Scripts Overview

Primary baseline pipeline:
- scripts/01_prepare_dataset.py: Parse VisDrone annotations and prepare YOLO data
- scripts/02_generate_weather.py: Generate weather-augmented sequences
- scripts/03_train_detector.py: Train baseline detector
- scripts/04_run_detection.py: Run inference (baseline and SAHI)
- scripts/05_run_tracking.py: Run MOT tracking on selected conditions
- scripts/06_evaluate.py: Evaluate MOT outputs and generate CSV results
- scripts/07_visualize.py: Build visual summaries and plots
- scripts/08_analysis.py: Generate analysis report artifacts

Extended and improved experiments:
- scripts/09_enhanced_videos.py: Higher-quality tracking video generation
- scripts/10_light_moderate_weather.py: Focused tracking/eval on light and moderate weather
- scripts/11_prepare_weather_training.py: Build weather-augmented training split
- scripts/12_train_weather_detector.py: Train weather-robust detector
- scripts/13_improved_pipeline.py: Improved pipeline (weather-trained model + BoT-SORT)
- scripts/14_compare_results.py: Compare old vs improved pipeline metrics
- scripts/15_ablation_bytetrack.py: Ablation with ByteTrack on improved detector
- scripts/16_compare_trackers.py: BoT-SORT vs ByteTrack comparison
- scripts/17_export_all_results.py: Export organized results for both trackers
- scripts/18_render_all_videos.py: Render complete output videos for report/demo

## Setup

### 1. Environment

Recommended:
- Python 3.8+
- CUDA-capable GPU for training/inference acceleration
- Windows/Linux with enough storage for generated outputs

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Required Assets

Ensure the following are present before running full experiments:
- VisDrone2019-MOT-train/ (dataset folder)
- pretrained/re-id weights as required by selected tracker settings (for example osnet_x0_25_msmt17.pt)
- YOLO checkpoints referenced by scripts (generated during training or provided externally)

## How To Run

### Baseline Flow

```bash
python scripts/01_prepare_dataset.py
python scripts/02_generate_weather.py
python scripts/03_train_detector.py
python scripts/04_run_detection.py
python scripts/05_run_tracking.py
python scripts/06_evaluate.py
python scripts/07_visualize.py
python scripts/08_analysis.py
```

### Improved Flow (Weather-Robust Model + Tracker Comparisons)

```bash
python scripts/11_prepare_weather_training.py
python scripts/12_train_weather_detector.py
python scripts/13_improved_pipeline.py
python scripts/14_compare_results.py
python scripts/15_ablation_bytetrack.py
python scripts/16_compare_trackers.py
python scripts/17_export_all_results.py
python scripts/18_render_all_videos.py
```

## Main Outputs

Common output directories:
- outputs/augmented/: weather-generated image sequences
- outputs/detections/: detector predictions by condition
- outputs/tracks_*/: tracker outputs in MOT format
- outputs/eval_results*/: per-experiment evaluation CSV files
- outputs/comparison/: tracker and pipeline comparison reports
- outputs/plots*/: summary plots for paper/report use
- outputs/videos*/: rendered qualitative result videos
- 
## Evaluation Metrics

The evaluation stages include standard MOT metrics:
- MOTA
- MOTP
- IDF1
- HOTA
- Precision
- Recall
- ID Switches
- FN (false negatives)

## Notes

- Script ordering is intentional; run numbered scripts in sequence unless you are resuming from an intermediate stage.
- Some scripts skip already generated artifacts to save runtime.
- For reproducible comparisons, keep tracker, detector, and weather condition definitions consistent across runs.
