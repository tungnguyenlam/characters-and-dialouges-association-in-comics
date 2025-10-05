
import os
import subprocess
import sys

def run_command(command):
    """Runs a shell command and prints its output."""
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)
    process.check_returncode()

# Cell 1: Potentially install specific ultralytics version
print("--- Cell 1: Checking ultralytics version ---")
# This was commented out, but we'll add a check and install if needed.
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Ultralytics not found. Installing...")
    run_command("pip install ultralytics==8.0.196") # A known stable version

try:
    import pycocotools
except ImportError:
    print("pycocotools not found. Installing...")
    run_command("pip install pycocotools")

# Cell 2: Change directory and setup
print("\n--- Cell 2: Setting up base directory ---")
# Assuming this script is run from the project root where it is created
if "characters-and-dialouges-association-in-comics" not in os.getcwd():
    # This is a fallback if not run from the root
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
print(f"Current directory: {os.getcwd()}")

BASE_DIR = os.getcwd()
END_WITH_LOCAL = 'characters-and-dialouges-association-in-comics'
if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    raise ValueError(f"Expected to be in .../{END_WITH_LOCAL} or .../content directory, but got: {BASE_DIR}")

# Cell 3: Imports and Model Loading
print("\n--- Cell 3: Imports and Model Loading ---")
import cv2
import numpy as np
import json
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

model = YOLO('yolov8x-seg.pt')
print("YOLOv8 model loaded.")

# Cell 4: Main Inference and Evaluation Logic
print("\n--- Cell 4: Main Inference and Evaluation Logic ---")
IMG_ROOT = os.path.join(BASE_DIR, "data/Manga109_released_2023_12_07/images")
JSON_ROOT = os.path.join(BASE_DIR, "data/MangaSegmentation/jsons")

output_dir = os.path.join(BASE_DIR, 'output', 'yolo_bubble_results')
os.makedirs(output_dir, exist_ok=True)

def get_pred_masks(result):
    pred_masks = []
    if result.masks is not None:
        # Assuming class 5 is 'bubble' or check by name
        for mask, cls in zip(result.masks.data, result.boxes.cls):
            class_name = result.names[int(cls)].lower()
            if class_name == 'speech bubble' or class_name == 'bubble': # Making it more robust
                pred_masks.append(mask.cpu().numpy())
    return pred_masks

def get_gt_masks_for_image(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[5]) # category_id 5 for bubble
    anns = coco.loadAnns(ann_ids)
    gt_masks = []
    for ann in anns:
        # The segmentation is a list of polygons
        if 'segmentation' in ann and ann['segmentation']:
            h, w = coco.imgs[img_id]['height'], coco.imgs[img_id]['width']
            rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
            gt_masks.append(maskUtils.decode(rle))
    return gt_masks

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

ious, precisions, recalls, f1s = [], [], [], []
manga_list = sorted(os.listdir(IMG_ROOT))

for manga_name in tqdm(manga_list, desc="Processing Mangas"):
    manga_dir = os.path.join(IMG_ROOT, manga_name)
    json_path = os.path.join(JSON_ROOT, f"{manga_name}.json")

    if not os.path.exists(json_path):
        print(f"⚠️ No JSON found for {manga_name}, skipping.")
        continue

    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc=f"Processing {manga_name}", leave=False):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(manga_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        results = model.predict(source=img_path, save=False, conf=0.25, iou=0.5, imgsz=640, verbose=False)
        result = results[0] # predict returns a list for a single image

        pred_masks = get_pred_masks(result)
        gt_masks = get_gt_masks_for_image(coco, img_id)

        tp = 0
        fp = len(pred_masks)
        fn = len(gt_masks)

        if not gt_masks and not pred_masks:
            continue # True negative, skip
        if not gt_masks:
            # All predictions are false positives
            pass
        elif not pred_masks:
            # All ground truths are false negatives
            pass
        else:
            # Match predictions to ground truths
            matched_gt_indices = set()
            for pred_mask in pred_masks:
                best_iou = 0
                best_gt_idx = -1
                for i, gt_mask in enumerate(gt_masks):
                    iou = compute_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                ious.append(best_iou)
                if best_iou > 0.5 and best_gt_idx not in matched_gt_indices:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)

        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

# Cell 5: Print Results
print("\n--- Cell 5: Printing Results ---")
if ious:
    print("\n🎯 Kết quả segmentation bubble:")
    print(f"👉 IoU trung bình:  {np.mean(ious):.3f}")
    print(f"👉 Precision:       {np.mean(precisions):.3f}")
    print(f"👉 Recall:          {np.mean(recalls):.3f}")
    print(f"👉 F1-score:        {np.mean(f1s):.3f}")
else:
    print("No bubbles were processed.")

# I am commenting out the visualization part as it requires a display
# and might not be suitable for a script. The main goal is to fix the execution.
# print("\n--- Cell 6: Visualizing Sample ---")
# import matplotlib.pyplot as plt
# if 'results' in locals() and results:
#     sample = results[0]
#     img = cv2.imread(sample.path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     for mask in get_pred_masks(sample):
#         img[mask > 0.5] = [255, 0, 0]
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title('Predicted Bubbles')
#     plt.savefig(os.path.join(output_dir, "sample_prediction.png"))
#     print(f"Sample image saved to {output_dir}")
