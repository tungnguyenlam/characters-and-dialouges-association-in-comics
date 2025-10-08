# YOLOv8 Balloon Detection Training Pipeline

A modular, production-ready pipeline for training YOLOv8 instance segmentation models on manga balloon detection.

## 📋 Overview

This pipeline splits the training workflow into 6 manageable steps with built-in checkpointing, validation, and error handling:

1. **Data Preparation** - Load and validate manga annotations
2. **Split and Prepare Dataset** - Create train/val splits in YOLO format
3. **Create YAML Configuration** - Generate dataset configuration
4. **Train Model** - Train YOLOv8 segmentation model
5. **Evaluate Model** - Run comprehensive evaluation
6. **Master Pipeline** - Orchestrate all steps

## 🚀 Quick Start

### Run Complete Pipeline

```bash
# Run all steps automatically
python pipeline.py
```

### Run Individual Steps

```bash
# Run steps one by one
python s1_2_data_preparation.py
python s3_split_prepare_yolo_dataset.py
python s4_create_yaml_config_file.py
python s5_train_model.py
python s6_eval_model.py
```

## 📁 File Structure

```
YOLOv8/
├── s1_2_data_preparation.py        # Steps 1-2: Data loading and filtering
├── s3_split_prepare_yolo_dataset.py # Step 3: Dataset splitting and conversion
├── s4_create_yaml_config_file.py   # Step 4: YAML configuration
├── s5_train_model.py               # Step 5: Model training
├── s6_eval_model.py                # Step 6: Model evaluation
├── pipeline.py                     # Master orchestrator
├── README_PIPELINE.md              # This file
└── .pipeline_state/                # Checkpoint directory (auto-created)
    ├── s1_2_complete.json          # Step 1-2 checkpoint
    ├── s3_complete.json            # Step 3 checkpoint
    ├── s4_complete.json            # Step 4 checkpoint
    └── s5_training_info.json       # Training metadata
```

## 📝 Pipeline Steps Details

### Step 1-2: Data Preparation

**Script:** `s1_2_data_preparation.py`

**What it does:**

- Validates required directories exist
- Loads processed JSON annotations
- Filters for balloon category (ID: 5)
- Creates structured data records

**Outputs:**

- `data_records.json` - Cached data for next steps
- `.pipeline_state/s1_2_complete.json` - Checkpoint

**Checkpointing:** ✅ Yes - Will skip if already completed

---

### Step 3: Split and Prepare YOLO Dataset

**Script:** `s3_split_prepare_yolo_dataset.py`

**What it does:**

- Groups images by manga series (prevents data leakage)
- Splits into train/val (80/20) by series
- Converts to YOLO instance segmentation format
- Copies images and creates label files

**Prerequisites:**

- Step 1-2 must be completed
- `data_records.json` must exist

**Outputs:**

- `data/YOLOv8_data/images/train/` - Training images
- `data/YOLOv8_data/images/val/` - Validation images
- `data/YOLOv8_data/labels/train/` - Training labels
- `data/YOLOv8_data/labels/val/` - Validation labels

**Checkpointing:** ✅ Yes - Detects partial work and cleans up

**Smart Features:**

- Detects incomplete work and automatically cleans up
- Verifies image/label count matches
- Validates checkpoint integrity

---

### Step 4: Create YAML Configuration

**Script:** `s4_create_yaml_config_file.py`

**What it does:**

- Creates `dataset.yaml` for YOLOv8
- Validates all paths are accessible
- Verifies class configuration

**Prerequisites:**

- Step 3 must be completed
- Train/val directories must contain files

**Outputs:**

- `data/YOLOv8_data/dataset.yaml` - Configuration file

**Checkpointing:** ✅ Yes - Validates existing YAML or regenerates

---

### Step 5: Train Model

**Script:** `s5_train_model.py`

**What it does:**

- Auto-detects hardware (MPS/CUDA/CPU)
- Loads pretrained YOLOv8 segmentation model
- Trains on prepared dataset
- Saves best model weights

**Prerequisites:**

- Step 4 must be completed
- `dataset.yaml` must exist

**Training Parameters:**

- Model: `yolov8s-seg.pt`
- Epochs: 5
- Image Size: 1280
- Batch Size: 4
- Device: Auto-detected

**Outputs:**

- `YOLOv8_Training_Results/balloon_segmentation_run1/`
  - `weights/best.pt` - Best model weights
  - `weights/last.pt` - Last epoch weights
  - `results.csv` - Training metrics
  - Various visualization plots

**Checkpointing:** ❌ No - Always runs fresh (allows retraining)

---

### Step 6: Evaluate Model

**Script:** `s6_eval_model.py`

**What it does:**

- Loads best trained model
- Runs validation on test set
- Generates comprehensive metrics report
- Creates performance visualizations

**Prerequisites:**

- Step 5 must be completed
- Trained model weights must exist

**Outputs:**

- Detailed console metrics report
- Bounding box metrics (mAP, precision, recall)
- Segmentation mask metrics
- Performance indicators

**Checkpointing:** ❌ No - Always runs fresh (allows re-evaluation)

---

## 🎯 Pipeline Orchestrator

### Basic Usage

```bash
# Run all steps
python pipeline.py

# Force re-run all steps (ignore checkpoints)
python pipeline.py --force

# Start from specific step
python pipeline.py --start-from 3

# Run specific steps only
python pipeline.py --steps 1 2 3
```

### Features

✅ **Automatic Checkpoint Management**

- Steps 1-4 use smart checkpointing
- Automatically skips completed steps
- Validates checkpoint integrity

✅ **Error Handling**

- Stops on first error
- Clear error messages
- Suggests remediation steps

✅ **Progress Tracking**

- Color-coded console output
- Step timing and duration
- Comprehensive summary report

✅ **Flexible Execution**

- Run all steps or specific ones
- Force re-execution
- Start from any step

## 🔧 Configuration

### Modify Training Parameters

Edit `s5_train_model.py`:

```python
# Training parameters
EPOCHS = 5              # Number of training epochs
IMAGE_SIZE = 1280       # Input image size
BATCH_SIZE = 4          # Batch size
PRETRAINED_MODEL = 'yolov8s-seg.pt'  # Base model
```

### Modify Target Category

Edit the respective files:

```python
TARGET_CATEGORY_ID = 5          # Balloon category ID
TARGET_CATEGORY_NAME = "balloon"  # Class name
```

## 📊 Understanding Checkpoints

### What Gets Checkpointed?

**Steps 1-4:** Full checkpointing with validation

- Saves completion status
- Stores output metadata
- Validates file checksums
- Detects partial work

**Steps 5-6:** No checkpointing (metadata only)

- Always run fresh
- Allows retraining/re-evaluation
- Training info saved for reference

### Checkpoint Files

```json
// Example: .pipeline_state/s1_2_complete.json
{
  "step": "s1_2",
  "timestamp": "2025-10-08T23:45:00",
  "status": "complete",
  "outputs": {
    "data_records_file": "data_records.json",
    "total_images": 1234,
    "total_annotations": 5678
  },
  "checksums": {
    "data_records": "sha256:abc123..."
  }
}
```

### Reset Pipeline

```bash
# Remove all checkpoints to start fresh
rm -rf .pipeline_state/
rm data_records.json

# Or use force mode
python pipeline.py --force
```

## 🐛 Troubleshooting

### "Step X has not been completed"

**Problem:** Trying to run a step without completing prerequisites

**Solution:**

```bash
# Run all previous steps first
python pipeline.py --start-from 1
```

### "Found partial work without valid checkpoint"

**Problem:** Previous run was interrupted

**Solution:** The pipeline automatically cleans up partial work. Just re-run the step.

### "No GPU detected. Using CPU"

**Problem:** No GPU available for training

**Solutions:**

- For Mac: Ensure you have Apple Silicon (M1/M2/M3)
- For CUDA: Install NVIDIA drivers and CUDA toolkit
- CPU training will work but be slower

### "Directory not found" errors

**Problem:** Required data directories missing

**Solution:**

```bash
# Ensure your data is in the correct location
# Expected structure:
# data/MangaSegmentation/jsons_processed/
# data/Manga109_released_2023_12_07/images/
```

## 📈 Performance Monitoring

### Training Progress

Monitor training in real-time:

- Console output shows progress
- TensorBoard logs (if enabled)
- Results saved to `YOLOv8_Training_Results/`

### Evaluation Metrics

Key metrics to watch:

- **mAP@50**: Mean Average Precision at IoU 0.5
- **mAP@50-95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

Performance indicators in evaluation report:

- 🟢 Excellent: ≥ 0.9
- 🟡 Good: ≥ 0.7
- 🟠 Fair: ≥ 0.5
- 🔴 Needs Improvement: < 0.5

## 💡 Tips and Best Practices

1. **Run Steps Individually First**

   - Easier to debug issues
   - Understand each step's output
   - Verify data quality at each stage

2. **Use Pipeline for Production**

   - Automated execution
   - Consistent results
   - Built-in error handling

3. **Monitor Disk Space**

   - YOLO dataset can be large
   - Training results include many plots
   - Clean up old runs if needed

4. **Backup Important Results**

   - Best model weights
   - Training configurations
   - Evaluation metrics

5. **Version Control**
   - Commit pipeline scripts
   - Track configuration changes
   - Document modifications

## 🔄 Workflow Examples

### Initial Training Run

```bash
# Complete pipeline from scratch
python pipeline.py
```

### Resume After Interruption

```bash
# Pipeline automatically detects completed steps
python pipeline.py
```

### Retrain with Different Parameters

```bash
# Edit s5_train_model.py (change EPOCHS, BATCH_SIZE, etc.)
# Run from step 5
python pipeline.py --start-from 5
```

### Evaluate Different Model

```bash
# Update model path in s6_eval_model.py if needed
python s6_eval_model.py
```

## 📞 Support

For issues or questions:

1. Check this README
2. Review error messages carefully
3. Verify prerequisites are met
4. Check checkpoint files for corruption

## 📄 License

See project root for license information.

---

**Happy Training! 🎉**
