# Google Colab Setup Instructions

## Quick Setup for Training

### 1. Clone Repository
```bash
cd /content
git clone https://github.com/tungnguyenlam/characters-and-dialouges-association-in-comics.git
cd characters-and-dialouges-association-in-comics
git checkout yolo
```

### 2. Upload Pretrained Weights (IMPORTANT!)

You have two options:

#### Option A: Manual Upload (Recommended for First Time)
Upload the pretrained weight files to the **root directory** of the repository:
- `yolo11n-seg.pt` (already in your local repo root)
- `yolo11s-seg.pt` (already in your local repo root)

In Colab, you can upload files using:
```python
from google.colab import files
uploaded = files.upload()  # Select yolo11n-seg.pt or yolo11s-seg.pt
```

Then move to root:
```bash
!mv yolo11n-seg.pt /content/characters-and-dialouges-association-in-comics/
```

#### Option B: Let Script Auto-Download (Requires Internet)
If you don't upload the weights, the script will attempt to download them automatically from Ultralytics. This requires:
- Stable internet connection
- May take a few minutes depending on connection speed
- The script now handles this automatically with improved error handling

### 3. Run the Training Pipeline
```bash
cd /content/characters-and-dialouges-association-in-comics
python code/bubble-detection/YOLO/run_pipeline.py
```

## Weight File Locations (Priority Order)

The script now searches for weights in this order:

1. **Root Directory**: `/content/characters-and-dialouges-association-in-comics/yolo11n-seg.pt`
2. **Current Working Directory**: Where the script is run from
3. **Auto-Download**: Downloads from Ultralytics if not found

## Troubleshooting

### If you see "FileNotFoundError: yolov11n-seg.pt"

**Solution 1**: Upload the weight file to root directory
```bash
# After uploading via Colab file upload
!mv yolo11n-seg.pt /content/characters-and-dialouges-association-in-comics/
```

**Solution 2**: Ensure internet connectivity for auto-download
```python
# Test internet connection
!ping -c 3 github.com
```

**Solution 3**: Download manually from Ultralytics
```bash
!wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
!mv yolo11n-seg.pt /content/characters-and-dialouges-association-in-comics/
```

### GPU Not Detected

Enable GPU in Colab:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 recommended)
3. Save

Verify GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Dataset Location

The script expects data in:
```
/content/characters-and-dialouges-association-in-comics/data/
├── Manga109_released_2023_12_07/
│   ├── images/
│   └── annotations/
└── YOLO_data/  (created by pipeline)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## Expected Output

After successful training, you'll find:
```
/content/characters-and-dialouges-association-in-comics/models/bubble-detection/yolo11n/
└── balloon_seg_run1/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    └── various plots
```

## Training Configuration (Current Settings)

- **Epochs**: 1 (for testing)
- **Batch Size**: 4
- **Image Size**: 640
- **Data Fraction**: 50% (using half the training data)
- **Optimizer**: AdamW
- **Learning Rate**: 0.001

To modify these, edit `s5_train_model.py` lines 73-77:
```python
EPOCHS = 1           # Increase for better training
BATCH_SIZE = 4       # Increase if you have more GPU memory
DATA_FRACTION = 0.5  # Set to 1.0 to use all training data
```

## Notes

- The improved script now handles multiple weight file locations
- Downloads are isolated in temporary directories to avoid conflicts
- Better error messages guide you if weights are not found
- All strategies are tried automatically before failing
