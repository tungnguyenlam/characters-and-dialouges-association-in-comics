#!/usr/bin/env python3
"""
Step 5: Train YOLO Model (Family Agnostic)
===========================================
This script handles:
- Validating that Step 4 was completed successfully
- Detecting available device (MPS/CUDA/CPU)
- Loading any pretrained YOLO family segmentation model
- Training the model on the prepared dataset
- Saving training results and model weights

Model Support:
- YOLOv8: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
- YOLOv9: yolov9-seg.pt, yolov9e-seg.pt
- YOLOv10: yolov10n-seg.pt, yolov10s-seg.pt, yolov10m-seg.pt, yolov10l-seg.pt, yolov10x-seg.pt
- YOLO11: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt

Usage:
- Simply change PRETRAINED_MODEL to any supported YOLO segmentation model
- The code automatically adapts folder structure and naming

Prerequisites:
- Step 4 must be completed successfully
- dataset.yaml must exist and be valid

Outputs:
- models/bubble-detection/{model_name}/: Training results
  - weights/best.pt: Best model weights
  - weights/last.pt: Last epoch weights
  - results.csv: Training metrics
  - Various plots and visualizations

Note: This step does NOT use checkpointing - it always runs fresh.
      This allows for retraining with different parameters if needed.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import re

# ===================================================================
# Configuration
# ===================================================================

END_WITH_LOCAL = 'characters-and-dialouges-association-in-comics'

BASE_DIR = os.getcwd()
print(f"BASE_DIR: {BASE_DIR}")

# Simple validation
if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    raise ValueError(f"Expected to be in .../{END_WITH_LOCAL} or .../content directory, but got: {BASE_DIR}")

# Paths
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'YOLO_data')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset.yaml')
PRETRAINED_MODEL = 'yolov8n-seg.pt'  # Change this to any YOLO model: yolo11n-seg.pt, yolov9-seg.pt, etc.

# Extract model base name for folder structure (removes -seg and .pt)
MODEL_BASE_NAME = re.sub(r'-seg\.pt$', '', PRETRAINED_MODEL)  # yolov8s-seg.pt -> yolov8s
MODEL_FAMILY = re.match(r'(yolo(?:v)?[\d]+)', MODEL_BASE_NAME, re.IGNORECASE)  # Extract yolov8, yolo11, etc.
MODEL_FAMILY_NAME = MODEL_FAMILY.group(1).upper() if MODEL_FAMILY else 'YOLO'  # YOLOv8, YOLO11, etc.

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection','YOLO', '.pipeline_state')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'models', 'bubble-detection', MODEL_FAMILY_NAME, 'weights')

# Training parameters
EPOCHS = 1
IMAGE_SIZE = 640
BATCH_SIZE = 1
PROJECT_NAME = os.path.join(BASE_DIR, 'models', 'bubble-detection', MODEL_BASE_NAME)

# Dynamic run name based on existing runs
def get_next_run_name(project_dir, base_name='run'):
    """Generate next available run name (run1, run2, etc.)"""
    if not os.path.exists(project_dir):
        return f"{base_name}1"
    
    existing_runs = [d for d in os.listdir(project_dir) 
                     if os.path.isdir(os.path.join(project_dir, d)) and d.startswith(base_name)]
    
    if not existing_runs:
        return f"{base_name}1"
    
    # Extract numbers from run names
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.replace(base_name, ''))
            run_numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(run_numbers) + 1 if run_numbers else 1
    return f"{base_name}{next_num}"

RUN_NAME = get_next_run_name(PROJECT_NAME, base_name='balloon_seg_run')

# ===================================================================
# Checkpoint System
# ===================================================================

def load_checkpoint(step_name):
    """Load checkpoint state. Returns None if not found or invalid."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{step_name}_complete.json")
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        if checkpoint.get("status") != "complete":
            return None
        
        return checkpoint
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}")
        return None

# ===================================================================
# Validation Functions
# ===================================================================

def check_prerequisites():
    """Check if Step 4 was completed successfully."""
    print("\n" + "="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Check for Step 4 checkpoint
    checkpoint = load_checkpoint("s4")
    if not checkpoint:
        raise RuntimeError(
            "❌ Step 4 has not been completed!\n"
            "Please run: python s4_create_yaml_config_file.py"
        )
    
    print("✓ Step 4 checkpoint found")
    print(f"  Timestamp: {checkpoint['timestamp']}")
    print(f"  YAML file: {checkpoint['outputs']['yaml_path']}")
    
    # Check YAML file exists
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(
            f"❌ YAML configuration file not found: {YAML_PATH}\n"
            "Please re-run Step 4: python s4_create_yaml_config_file.py"
        )
    
    print(f"✓ YAML configuration file found: {YAML_PATH}")
    
    # Check dataset directories
    train_img_dir = os.path.join(DATASET_DIR, 'images/train')
    val_img_dir = os.path.join(DATASET_DIR, 'images/val')
    
    if not os.path.exists(train_img_dir) or not os.listdir(train_img_dir):
        raise FileNotFoundError(f"❌ Training images not found or empty: {train_img_dir}")
    
    if not os.path.exists(val_img_dir) or not os.listdir(val_img_dir):
        raise FileNotFoundError(f"❌ Validation images not found or empty: {val_img_dir}")
    
    train_count = len(os.listdir(train_img_dir))
    val_count = len(os.listdir(val_img_dir))
    
    print(f"✓ Training images: {train_count}")
    print(f"✓ Validation images: {val_count}")
    
    print("\n✓ All prerequisites satisfied\n")
    
    return checkpoint

def detect_device():
    """Detect and configure the best available device for training."""
    print("\n" + "="*60)
    print("Detecting Hardware Device")
    print("="*60)
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ CUDA GPU detected: {gpu_name}")
        print(f"  Memory: {gpu_memory:.2f} GB")
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    #     print(f"\n✓ Apple Silicon (MPS) detected")
    #     print(f"  Metal Performance Shaders enabled")
    else:
        device = "cpu"
        print(f"\n⚠ No GPU detected. Using CPU")
        print(f"  Warning: Training will be significantly slower!")
    
    print(f"\n✓ Selected device: {device.upper()}")
    return device

# ===================================================================
# Main Training Function
# ===================================================================

def train_model(device):
    """Train the YOLO segmentation model."""
    print("\n" + "="*60)
    print(f"STEP 5: Training {MODEL_FAMILY_NAME} Model ({PRETRAINED_MODEL})")
    print("="*60)
    
    # Display training configuration
    print("\nTraining Configuration:")
    print(f"  Model: {PRETRAINED_MODEL}")
    print(f"  Dataset: {YAML_PATH}")
    print(f"  Device: {device.upper()}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Create weights directory if it doesn't exist
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Full path to model weights
    model_path = os.path.join(WEIGHTS_DIR, PRETRAINED_MODEL)
    
    # Load pretrained model
    print(f"\nLoading pretrained model: {model_path}")
    os.chdir(WEIGHTS_DIR)
    try:
        model = YOLO(PRETRAINED_MODEL)
        print("✓ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")
    os.chdir(BASE_DIR)
    # Start training
    # ...existing code...
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print("\nThis may take a while depending on your hardware.")
    print("Training progress will be displayed below:\n")
    
    try:
        results = model.train(
            data=str(YAML_PATH),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True,
            verbose=True,
            workers=2
        )
        
        print("\n" + "="*60)
        print("✓ Training Completed Successfully!")
        print("="*60)
        
        # Get training results location
        save_dir = Path(model.trainer.save_dir)
        best_model = Path(model.trainer.best)
        
        print(f"\nTraining results saved to: {save_dir}")
        print(f"Best model saved to: {best_model}")
        
        # Display key files
        print("\nGenerated files:")
        key_files = [
            'weights/best.pt',
            'weights/last.pt',
            'results.csv',
            'results.png',
            'confusion_matrix.png'
        ]
        
        for file in key_files:
            file_path = save_dir / file
            if file_path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  - {file} (not found)")
        
        return {
            "save_dir": str(save_dir),
            "best_model": str(best_model),
            "epochs_completed": EPOCHS
        }
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Training Failed!")
        print("="*60)
        raise RuntimeError(f"Training error: {e}")

# ===================================================================
# Main Execution
# ===================================================================

def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print(f"# Pipeline Step 5: Train {MODEL_FAMILY_NAME} Model")
    print("#"*60)
    
    print("\nNote: This step always runs fresh (no checkpointing).")
    print("This allows retraining with different parameters if needed.\n")
    
    # Check prerequisites
    check_prerequisites()
    
    # Detect device
    device = detect_device()
    
    # Train model
    training_info = train_model(device)
    
    print("\n" + "="*60)
    print("✓ Step 5 Complete!")
    print("="*60)
    print(f"\nModel training finished successfully!")
    print(f"Results directory: {training_info['save_dir']}")
    print(f"Best model: {training_info['best_model']}")
    print(f"\nReady for Step 6: Model evaluation")
    
    # Save training info for next step (not a checkpoint, just metadata)
    training_info_file = os.path.join(CHECKPOINT_DIR, 's5_training_info.json')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(training_info_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "training_info": training_info,
            "device": device,
            "epochs": EPOCHS,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE
        }, f, indent=2)
    print(f"\nTraining metadata saved to: {training_info_file}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 5: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
