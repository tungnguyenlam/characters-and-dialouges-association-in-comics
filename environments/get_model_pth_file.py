import os
import gdown

BASE_DIR = os.getcwd()
if not BASE_DIR.endswith('group-project-b3'):
    raise ValueError('To run this you should be in the group-project-b3 directory')

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'bubble-detection')

model_dir_dict = {
    'yolo_scratch': os.path.join(MODEL_DIR, 'model-scratch-manga-segmentation', 'yolo_outputs'),
    'detectron2': os.path.join(MODEL_DIR, 'detectron2', 'output_balloon_segmentation_v3'),
    'deeplabv3': os.path.join(MODEL_DIR, 'DEEPLABv3'),
    'unet': os.path.join(MODEL_DIR, 'UNET'),
    'yolo_ultralytics': os.path.join(MODEL_DIR, 'YOLO_Ultralytics'),
}


path_file_dict = {
    'yolo_scratch': "https://drive.google.com/file/d/1pCSfEszGmHxmkxUwqNOz2A9bOoSSzpi4/view",
    'detectron2': "https://drive.google.com/file/d/1JXJ99PFviYwB4PENSkDUcSuzIMy65qUi/view",
    'deeplabv3': "https://drive.google.com/file/d/1DqqjKqS8Sgc08TWfjkApU3yumHJq37fP/view",
    'unet': "https://drive.google.com/file/d/107dMAAXMIk3WNWLxENAZpiy7lrovEoXZ/view",

    'yolov8n': "https://drive.google.com/file/d/1VXrto3UwTikI02rBMh21K9jFmtPjxHsR/view?usp=sharing",
    'yolov8s': "https://drive.google.com/file/d/14-Nedp4C-bFENgpNFNijK3Oj0TV6Axka/view?usp=sharing",
    'yolo11n': "https://drive.google.com/file/d/1m0jUSxA8GzaPGZJqA8o2Q8KR3yJOFfdc/view?usp=sharing",
    'yolo11s': "https://drive.google.com/file/d/1tgP3lTr_zjVEsa8zMZ_nYUusc3K5CzY2/view?usp=sharing",
}

output_filename_dict = {
    'yolo_scratch': 'yolo_best.pth',
    'detectron2': 'model_final.pth',
    'deeplabv3': 'best_deeplabv3_model.pth',
    'unet': 'best_unet_model.pth',

    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolo11n': 'yolov11n.pt',
    'yolo11s': 'yolov11s.pt',
}


def download_if_not_exists(url, output_path, name):
    if os.path.exists(output_path):
        print(f"[SKIP] {name} already exists: {output_path}")
        return
    print(f"Downloading {name}...")
    gdown.download(url=url, output=output_path, fuzzy=True, quiet=False)


for key in ['yolo_scratch', 'detectron2', 'deeplabv3', 'unet']:
    dst_dir = model_dir_dict[key]
    os.makedirs(dst_dir, exist_ok=True)
    output_file = os.path.join(dst_dir, output_filename_dict[key])
    download_if_not_exists(path_file_dict[key], output_file, key)

yolo_ultra_dir = model_dir_dict['yolo_ultralytics']
os.makedirs(yolo_ultra_dir, exist_ok=True)

for key in ['yolov8n', 'yolov8s', 'yolo11n', 'yolo11s']:
    output_file = os.path.join(yolo_ultra_dir, output_filename_dict[key])
    download_if_not_exists(path_file_dict[key], output_file, key)


