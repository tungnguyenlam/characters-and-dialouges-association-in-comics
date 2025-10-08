#!/usr/bin/env python3
"""
Quick start

- Make sure you are in **.../characters-and-dialouges-association-in-comics/**
- Activate the conda environment
- Run

```bash
python ./code/utls/rle_to_polygons.py
python ./code/bubble-detection/YOLOv8/pipeline.py
```

# PIPELINE SUMMARY

## Created Files:

1. s1_2_data_preparation.py - Steps 1-2: Data loading and filtering
2. s3_split_prepare_yolo_dataset.py - Step 3: Dataset splitting and conversion
3. s4_create_yaml_config_file.py - Step 4: YAML configuration
4. s5_train_model.py - Step 5: Model training
5. s6_eval_model.py - Step 6: Model evaluation
6. pipeline.py - Master orchestrator
7. README_PIPELINE.md - Complete documentation

## Quick Start:

Option 1: Run Complete Pipeline
python pipeline.py

Option 2: Run Steps Individually
python s1_2_data_preparation.py
python s3_split_prepare_yolo_dataset.py
python s4_create_yaml_config_file.py
python s5_train_model.py
python s6_eval_model.py

## Pipeline Features:

✓ Smart Checkpointing (Steps 1-4)

- Automatically skips completed steps
- Detects and cleans partial work
- Validates file integrity

✓ Prerequisite Validation

- Each step checks previous steps
- Clear error messages
- Suggests remediation

✓ Device Auto-Detection

- MPS (Apple Silicon)
- CUDA (NVIDIA GPU)
- CPU fallback

✓ Comprehensive Error Handling

- Graceful failure
- Detailed error messages
- Safe cleanup

✓ Progress Tracking

- Color-coded console output
- Step timing
- Summary reports

## Advanced Usage:

Force re-run all steps:
python pipeline.py --force

Start from specific step:
python pipeline.py --start-from 3

Run specific steps only:
python pipeline.py --steps 1 2 3

## Directory Structure:

YOLOv8/
├── s1_2_data_preparation.py
├── s3_split_prepare_yolo_dataset.py
├── s4_create_yaml_config_file.py
├── s5_train_model.py
├── s6_eval_model.py
├── pipeline.py
├── README_PIPELINE.md
└── .pipeline_state/ (auto-created)
├── s1_2_complete.json
├── s3_complete.json
├── s4_complete.json
└── s5_training_info.json

## Output Structure:

data/
├── YOLOv8_data/ (Created by Step 3)
│ ├── images/
│ │ ├── train/
│ │ └── val/
│ ├── labels/
│ │ ├── train/
│ │ └── val/
│ └── dataset.yaml (Created by Step 4)
│
YOLOv8_Training_Results/ (Created by Step 5)
└── balloon_segmentation_run1/
├── weights/
│ ├── best.pt
│ └── last.pt
├── results.csv
├── results.png
└── [various plots]

## Key Differences from Original:

1. Modular Design

   - Single monolithic file → 6 focused scripts
   - Each step has clear responsibility
   - Easier to debug and maintain

2. Checkpointing System

   - Steps 1-4 save completion state
   - Automatic resume after interruption
   - Validates file integrity

3. Smart Validation

   - Each step checks prerequisites
   - Detects partial/corrupted work
   - Automatic cleanup and restart

4. Better Error Handling

   - Clear error messages
   - Suggests fixes
   - Graceful failure

5. Pipeline Orchestrator

   - Run all steps automatically
   - Flexible execution options
   - Progress tracking and summary

6. Device Detection
   - Fixed MPS support for Apple Silicon
   - Auto-detects CUDA/MPS/CPU
   - Properly passes device to training

## Benefits:

✓ Reliability

- Checkpoint system prevents data loss
- Validates each step before proceeding
- Detects and fixes partial work

✓ Flexibility

- Run any step independently
- Retrain without reprocessing data
- Easy to modify parameters

✓ Maintainability

- Clear separation of concerns
- Each script is focused
- Easy to update individual steps

✓ Usability

- Simple command-line interface
- Color-coded output
- Comprehensive documentation

✓ Production Ready

- Robust error handling
- Logging and monitoring
- Safe for automation

## Troubleshooting:

See README_PIPELINE.md for detailed troubleshooting guide

Common Issues:

1. "Step X not completed" → Run previous steps first
2. "Partial work detected" → Automatic cleanup (just re-run)
3. "No GPU detected" → Normal for CPU-only systems
4. "Directory not found" → Check data structure

## Reset Everything:

    rm -rf .pipeline_state/
    rm data_records.json
    python pipeline.py

## For More Information:

See README_PIPELINE.md for complete documentation
"""

print(**doc**)

```

```
