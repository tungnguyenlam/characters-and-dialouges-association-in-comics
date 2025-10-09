#!/usr/bin/env python3
"""
Step 5: Train YOLO Model (Family Agnostic) - INDEPENDENT VERSION
=================================================================
This script handles:
- Detecting available device (MPS/CUDA/CPU)
- Loading any pretrained YOLO family segmentation model
- Training the model on the prepared dataset
- Saving training results and model weights

Model Support:
- YOLOv8: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
- YOLOv10: yolov10n-seg.pt, yolov10s-seg.pt, yolov10m-seg.pt, yolov10l-seg.pt, yolov10x-seg.pt
- YOLO11: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt

Usage:
- Simply change PRETRAINED_MODEL to any supported YOLO segmentation model
- The code automatically adapts folder structure and naming
- Can run independently without completing previous steps

Prerequisites:
- dataset.yaml must exist and be valid (run s4 once, then you can skip it)

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
import re
import json
import torch
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import shutil
import yaml

from typing import List, Optional, Sequence, Set

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
PRETRAINED_MODEL = 'yolov11n-seg.pt'  # Change this to any YOLO model

# Extract model base name for folder structure (removes -seg and .pt)
MODEL_BASE_NAME = re.sub(r'-seg\.pt$', '', PRETRAINED_MODEL)  # yolov8s-seg.pt -> yolov8s
MODEL_FAMILY = re.match(r'(yolo(?:v)?[\d]+)', MODEL_BASE_NAME, re.IGNORECASE)  # Extract yolov8, yolo11, etc.
MODEL_FAMILY_NAME = MODEL_FAMILY.group(1).upper() if MODEL_FAMILY else 'YOLO'  # YOLOv8, YOLO11, etc.

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection','YOLO', '.pipeline_state')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'models', 'bubble-detection', MODEL_FAMILY_NAME, 'weights')
MIN_WEIGHT_BYTES = 1 * 1024 * 1024  # 1 MB sanity check for pretrained weights

# Training parameters
EPOCHS = 1
IMAGE_SIZE = 640
BATCH_SIZE = 4
DATA_FRACTION = 0.5  # Use 50% of training data

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
        match = re.search(r'(\d+)$', run)
        if match:
            run_numbers.append(int(match.group(1)))
    
    next_num = max(run_numbers) + 1 if run_numbers else 1
    return f"{base_name}{next_num}"

RUN_NAME = get_next_run_name(PROJECT_NAME, base_name='balloon_seg_run')

# ===================================================================
# Validation Functions (SIMPLIFIED)
# ===================================================================

def check_prerequisites():
    """Check if dataset.yaml exists (minimal prerequisite check)."""
    print("\n" + "="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Only check YAML file - the ONE thing we actually need
    if not os.path.exists(YAML_PATH):
        print("\n❌ YAML configuration file not found!")
        print(f"Expected: {YAML_PATH}")
        print("\nYou need to run Step 4 at least once:")
        print("  python s4_create_yaml_config_file.py")
        print("\nAfter that, you can run Step 5 independently anytime.")
        raise FileNotFoundError(f"Missing dataset.yaml at {YAML_PATH}")
    
    print(f"✓ YAML configuration file found: {YAML_PATH}")
    
    # Verify YAML content
    try:
        import yaml
        with open(YAML_PATH, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        missing = [field for field in required_fields if field not in yaml_content]
        
        if missing:
            raise ValueError(f"YAML missing required fields: {missing}")
        
        print(f"✓ YAML validated: {yaml_content['nc']} classes")
        print(f"  Classes: {yaml_content['names']}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not validate YAML content: {e}")
        print("  Proceeding anyway...")
    
    # Check if dataset directories exist (informational only)
    train_img_dir = os.path.join(DATASET_DIR, 'images/train')
    val_img_dir = os.path.join(DATASET_DIR, 'images/val')
    
    if os.path.exists(train_img_dir) and os.path.exists(val_img_dir):
        train_count = len([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
        val_count = len([f for f in os.listdir(val_img_dir) if f.endswith('.jpg')])
        print(f"\n✓ Dataset found:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
    else:
        print("\n⚠ Warning: Dataset directories not found!")
        print("  Training will fail if dataset is not prepared.")
    
    print("\n✓ Prerequisites check complete\n")

def detect_device():
    """Detect and return the best available device."""
    print("\n" + "="*60)
    print("Detecting Hardware")
    print("="*60)
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✓ Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        print("⚠ No GPU detected, using CPU")
        print("  Note: Training will be SLOW on CPU!")
    
    print(f"\nUsing device: {device.upper()}")
    return device

# ===================================================================
# Main Training Function
# ===================================================================

def _backup_corrupted_weights(weight_path: Path) -> None:
    """Backup a potentially corrupted weight file."""
    backup_path = weight_path.with_suffix('.pt.backup')
    print(f"  Backing up to: {backup_path}")
    shutil.move(str(weight_path), str(backup_path))

def _candidate_weight_names():
    """Generate list of candidate weight filenames to check."""
    base = PRETRAINED_MODEL.replace('-seg', '')
    names = {
        PRETRAINED_MODEL,
        PRETRAINED_MODEL.lower(),
        PRETRAINED_MODEL.upper(),
        base,
        base.lower(),
        base.upper(),
        f"{base}-seg.pt",
        f"{base}.pt",
    }
    return [name for name in names if name]


def _parse_additional_dirs(value: Optional[str]) -> List[Path]:
    """Parse colon/semicolon separated list of directories from environment variable."""
    if not value:
        return []
    dirs: List[Path] = []
    for item in value.split(os.pathsep):
        if not item:
            continue
        path = Path(item).expanduser()
        dirs.append(path)
    return dirs


def _unique_existing_dirs(paths: Sequence[Path]) -> List[Path]:
    """Deduplicate and keep only existing directories, preserving order."""
    unique: List[Path] = []
    seen: Set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if not path.exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _match_weight_file(path: Path) -> bool:
    """Return True if path looks like the desired pretrained weight."""
    if not path.is_file():
        return False

    filename = path.name.lower()
    targets = {name.lower() for name in _candidate_weight_names()}
    if filename in targets:
        return True

    # Heuristic: ensure base model name appears in filename for unexpected variants
    base = PRETRAINED_MODEL.replace('.pt', '').replace('-seg', '').lower()
    return base in filename and filename.endswith('.pt')


def _locate_existing_weight() -> Optional[Path]:
    """Attempt to locate an existing pretrained weight file."""
    print("\n🔍 Searching for existing pretrained weights...")

    env_path = os.environ.get('YOLO_PRETRAINED_PATH')
    if env_path:
        candidate = Path(env_path).expanduser()
        print(f"  Checking YOLO_PRETRAINED_PATH={candidate}")
        if _match_weight_file(candidate):
            print(f"✓ Found via YOLO_PRETRAINED_PATH: {candidate}")
            return candidate
        else:
            print("  ⚠ Path from YOLO_PRETRAINED_PATH is invalid or not found")

    additional_dirs = _parse_additional_dirs(os.environ.get('YOLO_WEIGHTS_DIRS'))

    common_dirs = [
        Path(WEIGHTS_DIR),
        Path(WEIGHTS_DIR).parent,
        Path(BASE_DIR),
        Path(BASE_DIR) / 'models',
        Path(BASE_DIR) / 'weights',
        Path(BASE_DIR) / 'pretrained',
        Path.cwd(),
        Path(BASE_DIR).parent,
        Path('/content') if Path('/content').exists() else None,
        Path.home(),
    ]

    search_dirs = _unique_existing_dirs([
        *(path for path in common_dirs if path),
        *additional_dirs,
    ])

    # Direct check for each candidate filename in directories
    candidate_names = _candidate_weight_names()
    for directory in search_dirs:
        for name in candidate_names:
            candidate = directory / name
            if _match_weight_file(candidate):
                print(f"✓ Found weights: {candidate}")
                return candidate

    # Recursive search as last resort
    for directory in search_dirs:
        try:
            print(f"  Scanning {directory} for {PRETRAINED_MODEL}...")
            for match in directory.rglob('*.pt'):
                if _match_weight_file(match):
                    print(f"✓ Found weights via scan: {match}")
                    return match
        except (OSError, PermissionError) as e:
            print(f"  ⚠ Skipping {directory}: {e}")
            continue

    print("  ✗ No existing weights located in known directories")
    return None

def _download_pretrained_weights(destination: Path):
    """Download pretrained weights from GitHub releases."""
    print(f"\n📥 Attempting to download from GitHub releases...")
    print(f"  This requires internet connectivity...")
    
    # Ultralytics hosts YOLO models on GitHub releases
    # URL pattern: https://github.com/ultralytics/assets/releases/download/v{version}/{model}
    # For YOLO11, version is typically v8.3.0 or later
    
    # Try multiple possible URLs
    possible_urls = [
        f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{PRETRAINED_MODEL}",
        f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{PRETRAINED_MODEL}",
    ]
    
    downloaded = False
    for url in possible_urls:
        try:
            print(f"    Trying: {url}")
            urllib.request.urlretrieve(url, str(destination))
            
            # Verify download
            if destination.exists() and destination.stat().st_size >= MIN_WEIGHT_BYTES:
                print(f"✓ Weights downloaded successfully")
                print(f"  Size: {destination.stat().st_size / 1024**2:.2f} MB")
                print(f"  Saved to: {destination}")
                
                # Load the model
                model = YOLO(str(destination))
                print(f"✓ Model loaded successfully")
                downloaded = True
                return model
            else:
                if destination.exists():
                    destination.unlink()  # Remove incomplete file
        except urllib.error.HTTPError as e:
            print(f"    ✗ Not found (HTTP {e.code})")
            continue
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    if not downloaded:
        raise RuntimeError("Could not download from any known URL")


def ensure_pretrained_model():
    """Ensure pretrained model is available and valid."""
    print("\n" + "="*60)
    print(f"Loading Pretrained Model: {PRETRAINED_MODEL}")
    print("="*60)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    existing_path = _locate_existing_weight()

    if existing_path:
        file_size = existing_path.stat().st_size
        print(f"  Size: {file_size / 1024**2:.2f} MB")

        if file_size < MIN_WEIGHT_BYTES:
            print(f"  ⚠ File too small (<1MB), treating as corrupted: {existing_path}")
        else:
            try:
                model = YOLO(str(existing_path))
                print(f"✓ Model loaded successfully from {existing_path}")

                # Ensure a copy exists in WEIGHTS_DIR for reproducibility
                dest_path = Path(WEIGHTS_DIR) / PRETRAINED_MODEL
                if existing_path.resolve() != dest_path.resolve():
                    try:
                        os.makedirs(dest_path.parent, exist_ok=True)
                        shutil.copy2(existing_path, dest_path)
                        print(f"✓ Copied weights to: {dest_path}")
                    except Exception as copy_err:
                        print(f"  ⚠ Failed to copy weights to {dest_path}: {copy_err}")

                return model
            except Exception as e:
                print(f"  ⚠ Failed to load model from {existing_path}: {e}")

    print(f"⚠ No usable pretrained weights found locally. Attempting download...")

    # No weights found anywhere, try to download
    try:
        return _download_pretrained_weights(Path(WEIGHTS_DIR) / PRETRAINED_MODEL)
    except Exception as e:
        error_msg = f"Failed to download weights: {e}"
        print(f"\n❌ {error_msg}")
        print(f"\n💡 Suggestions:")
        print(f"  1. Verify the file exists: ls -lh {BASE_DIR}/{PRETRAINED_MODEL}")
        print(f"  2. Upload or move the file, then set YOLO_PRETRAINED_PATH=/full/path/to/{PRETRAINED_MODEL}")
        print(f"  3. Provide additional search directories via YOLO_WEIGHTS_DIRS")
        print("  4. Manual download example:")
        print(f"     !wget https://github.com/ultralytics/assets/releases/download/v8.3.0/{PRETRAINED_MODEL} \\")
        print(f"          -O {Path(WEIGHTS_DIR) / PRETRAINED_MODEL}")
        raise RuntimeError(error_msg)

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
    print(f"  Data Fraction: {DATA_FRACTION * 100:.0f}%")
    print(f"  Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Ensure pretrained weights are available and loadable
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = Path(WEIGHTS_DIR) / PRETRAINED_MODEL

    print(f"\nLoading pretrained model: {model_path}")
    try:
        model = ensure_pretrained_model()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare pretrained model: {e}")
    
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
            workers=1,
            
            # ============ Optimizer & Learning Rate ============
            optimizer='AdamW',      # Better for short training runs than SGD
            lr0=0.001,              # Lower initial LR for stability in 1 epoch
            lrf=0.1,                # Final LR will be 0.0001
            momentum=0.9,           # Slightly lower for single epoch
            weight_decay=0.0005,
            warmup_epochs=0.0,      # No warmup needed for 1 epoch
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=False,           # Linear LR decay fine for 1 epoch
            
            # ============ Loss Weights (Balanced) ============
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # ============ Data Augmentation (AGGRESSIVE) ============
            # Geometric Augmentations
            mosaic=1.0,             # Keep mosaic for better feature learning
            close_mosaic=0,         # Don't close mosaic (only 1 epoch)
            mixup=0.15,             # Moderate mixup for regularization
            copy_paste=0.3,         # Good for segmentation tasks
            degrees=15.0,           # Increased rotation
            translate=0.2,          # Increased translation
            scale=0.7,              # Increased scale variation
            shear=5.0,              # Added shear
            perspective=0.0005,     # Added perspective distortion
            fliplr=0.5,             # Keep horizontal flip
            flipud=0.0,             # No vertical flip for speech bubbles
            
            # Color Augmentations
            hsv_h=0.02,             # Slightly increased hue
            hsv_s=0.7,              # Keep saturation
            hsv_v=0.4,              # Keep brightness
            auto_augment='randaugment',  # Keep auto augmentation
            erasing=0.4,            # Random erasing for robustness
            bgr=0.0,                # No BGR swap
            
            # ============ Performance & Memory ============
            cache=False,            # No caching (saves disk space)
            amp=True,               # Automatic Mixed Precision
            multi_scale=False,      # Disable for faster training
            rect=False,             # Disable rectangular training
            deterministic=False,    # Allow non-deterministic for speed
            compile=False,          # Avoid compilation overhead for 1 epoch
            half=False,             # AMP handles precision
            
            # ============ Regularization (Anti-Overfitting) ============
            dropout=0.15,           # Add dropout for regularization
            overlap_mask=True,
            mask_ratio=4,
            retina_masks=False,
            
            # ============ Validation & Monitoring ============
            val=True,
            plots=True,
            save=True,
            save_period=-1,         # Only save best and last
            patience=100,           # Not relevant for 1 epoch
            conf=None,
            iou=0.7,
            max_det=300,
            single_cls=False,
            
            # ============ Other ============
            fraction=DATA_FRACTION, # Use specified fraction of data
            seed=42,                # For reproducibility
            pretrained=True,
            freeze=None,            # Don't freeze any layers
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
    
    print("\n✨ INDEPENDENT MODE: This script can run without completing previous steps")
    print("   (as long as dataset.yaml exists)\n")
    
    # Check prerequisites (only YAML file)
    check_prerequisites()
    
    # Detect device
    device = detect_device()
    
    # Train model
    training_info = train_model(device)
    
    print("\n" + "="*60)
    print("✓ Step 5 Complete!")
    print("="*60)
    print(f"\nModel saved to: {training_info['best_model']}")
    print(f"\nYou can now:")
    print(f"  1. Run evaluation: python s6_eval_model.py")
    print(f"  2. Retrain with different parameters (just edit and re-run this script)")
    print(f"  3. Use the model for inference")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 5: {e}")
        import traceback
        traceback.print_exc()
        exit(1)