#!/usr/bin/env python3
"""
Step 1-2: Data Preparation Pipeline
====================================
This script handles:
- Directory validation and setup
- Loading and parsing processed JSON files
- Filtering for balloon annotations
- Creating data records for downstream processing

Outputs:
- data_records.json: Cached data records for next step
- .pipeline_state/s1_2_complete.json: Checkpoint file
"""

import os
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ===================================================================
# Configuration
# ===================================================================

END_WITH_LOCAL = 'characters-and-dialouges-association-in-comics'

os.environ['PATH'] = f"/root/.cargo/bin:{os.environ['PATH']}"

BASE_DIR = os.getcwd()
print(f"BASE_DIR: {BASE_DIR}")

# Simple validation
if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    raise ValueError(f"Expected to be in .../{END_WITH_LOCAL} or .../content directory, but got: {BASE_DIR}")

# Paths
JSON_DIR = os.path.join(BASE_DIR, 'data', 'MangaSegmentation/jsons_processed')
IMAGE_ROOT_DIR = os.path.join(BASE_DIR, 'data', 'Manga109_released_2023_12_07/images')
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'YOLO_data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', '.pipeline_state')
DATA_RECORDS_FILE = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', 'data_records.json')

# Target category
TARGET_CATEGORY_ID = 5  # Fixed category ID for balloon
TARGET_CATEGORY_NAME = "balloon"  # Fixed category name

# ===================================================================
# Checkpoint System
# ===================================================================

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_checkpoint(step_name, outputs, checksums=None):
    """Save checkpoint state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "status": "complete",
        "outputs": outputs,
        "checksums": checksums or {}
    }
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{step_name}_complete.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"✓ Checkpoint saved: {checkpoint_file}")

def load_checkpoint(step_name):
    """Load checkpoint state. Returns None if not found or invalid."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{step_name}_complete.json")
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        # Validate checkpoint structure
        if checkpoint.get("status") != "complete":
            return None
        
        # Verify output files exist
        data_records_path = checkpoint.get("outputs", {}).get("data_records_file")
        if data_records_path and os.path.exists(data_records_path):
            # Verify checksum if available
            saved_checksum = checkpoint.get("checksums", {}).get("data_records")
            if saved_checksum:
                current_checksum = calculate_file_hash(data_records_path)
                if current_checksum != saved_checksum:
                    print("⚠ Checkpoint file corrupted (checksum mismatch). Will regenerate.")
                    return None
            return checkpoint
        else:
            print("⚠ Output files from checkpoint not found. Will regenerate.")
            return None
            
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}. Will regenerate.")
        return None

# ===================================================================
# Main Functions
# ===================================================================

def validate_directories():
    """Validate that required directories exist and contain data."""
    print("\n" + "="*60)
    print("STEP 1: Validating Directories")
    print("="*60)
    
    # Ensure output directories exist
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Validate paths
    for path_name, path in [("JSON Directory", JSON_DIR), ("Image Root Directory", IMAGE_ROOT_DIR)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")
        
        contents = os.listdir(path)
        if not contents:
            raise ValueError(f"{path_name} is empty: {path}")
        
        print(f"✓ Found {path_name}: {path}")
        print(f"  Sample contents: {contents[:5]}")
    
    # Validate JSON files exist
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    if not json_files:
        raise ValueError(f"No JSON files found in {JSON_DIR}")
    
    print(f"✓ Found {len(json_files)} JSON files to process")
    print("\n✓ Directory validation complete\n")

def prepare_manga_balloon_data(json_dir, image_root):
    """
    Load pre-processed JSON files, filter for target category,
    and return a list of image records.
    """
    print("\n" + "="*60)
    print("STEP 2: Preparing Data from Processed JSONs")
    print("="*60)
    
    all_images = {}
    all_annotations = defaultdict(list)

    print("\nLoading and parsing JSON files...")
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSONs"):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                for img_info in data.get('images', []):
                    all_images[img_info['id']] = img_info
                for ann_info in data.get('annotations', []):
                    all_annotations[ann_info['image_id']].append(ann_info)
        except Exception as e:
            print(f"⚠ Error processing {json_file}: {e}")
            continue

    print(f"\n✓ Loaded data for {len(all_images)} total images")

    dataset_records = []
    images_with_balloons = 0
    total_balloon_annotations = 0
    
    for img_id, img_info in tqdm(all_images.items(), desc="Filtering balloon annotations"):
        # Create a base record for the image
        record = {
            "file_name": os.path.join(image_root, img_info['file_name']),
            "image_id": img_id,
            "height": img_info['height'],
            "width": img_info['width'],
        }
        
        # Filter for balloon annotations
        balloon_annotations = []
        for ann in all_annotations.get(img_id, []):
            if ann.get('category_id') == TARGET_CATEGORY_ID:
                # Ensure segmentation data is present and not empty
                if ann.get('segmentation'):
                    balloon_annotations.append({
                        "segmentation": ann['segmentation'],
                        "category_id": 0,  # All balloons will be class 0
                    })
                    total_balloon_annotations += 1
        
        # Only add images that contain at least one balloon
        if balloon_annotations:
            record["annotations"] = balloon_annotations
            dataset_records.append(record)
            images_with_balloons += 1
    
    print(f"\n✓ Data preparation complete:")
    print(f"  - Total images with '{TARGET_CATEGORY_NAME}': {images_with_balloons}")
    print(f"  - Total '{TARGET_CATEGORY_NAME}' annotations: {total_balloon_annotations}")
    
    if images_with_balloons == 0:
        raise ValueError(f"No images found containing '{TARGET_CATEGORY_NAME}' annotations!")
    
    return dataset_records, {
        "total_images": images_with_balloons,
        "total_annotations": total_balloon_annotations
    }

def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("# Pipeline Step 1-2: Data Preparation")
    print("#"*60)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint("s1_2")
    if checkpoint:
        print("\n✓ Step 1-2 already completed!")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  Total images: {checkpoint['outputs']['total_images']}")
        print(f"  Total annotations: {checkpoint['outputs']['total_annotations']}")
        print(f"  Data file: {checkpoint['outputs']['data_records_file']}")
        print("\nSkipping to next step...")
        return
    
    print("\nNo valid checkpoint found. Starting fresh...\n")
    
    # Step 1: Validate directories
    validate_directories()
    
    # Step 2: Prepare data
    dataset_records, stats = prepare_manga_balloon_data(JSON_DIR, IMAGE_ROOT_DIR)
    
    # Save data records
    print("\nSaving data records...")
    with open(DATA_RECORDS_FILE, 'w') as f:
        json.dump(dataset_records, f, indent=2)
    print(f"✓ Data records saved to: {DATA_RECORDS_FILE}")
    
    # Calculate checksum
    data_checksum = calculate_file_hash(DATA_RECORDS_FILE)
    
    # Save checkpoint
    save_checkpoint(
        "s1_2",
        outputs={
            "data_records_file": DATA_RECORDS_FILE,
            "total_images": stats["total_images"],
            "total_annotations": stats["total_annotations"]
        },
        checksums={
            "data_records": data_checksum
        }
    )
    
    print("\n" + "="*60)
    print("✓ Step 1-2 Complete!")
    print("="*60)
    print(f"\nTarget Category: {TARGET_CATEGORY_NAME} (ID: {TARGET_CATEGORY_ID})")
    print(f"Images with balloons: {stats['total_images']}")
    print(f"Total balloon annotations: {stats['total_annotations']}")
    print(f"\nReady for Step 3: Dataset splitting and YOLO format conversion")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 1-2: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
