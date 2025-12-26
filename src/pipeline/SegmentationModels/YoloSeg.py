from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import numpy as np
from typing import Tuple, List, Any
from .BaseSegmentationModel import BaseSegmentationModel


class YoloSeg(BaseSegmentationModel):
    def __init__(
        self, 
        model_path: str, 
        device: str = 'auto', 
        verbose: bool = False,
        plot: bool = False
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            verbose=verbose,
            plot=plot
        )

    def load_model(self) -> None:
        """Load YOLO segmentation model."""
        self.model = YOLO(self.model_path)
        self._log("Model loaded successfully")

    def _inference(
        self, 
        image: np.ndarray, 
        conf_threshold: float = 0.5,
        **kwargs
    ) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
        """Core inference logic for YOLO segmentation."""
        results = self.model.predict(
            image, 
            device=self.device, 
            verbose=self.verbose,
            conf=conf_threshold
        )
        result = results[0]
        
        # Extract bounding boxes
        bboxes = result.boxes.xyxy.cpu().numpy().astype(float).tolist()
        
        # Extract masks
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        
        # Extract confidence scores
        confs = result.boxes.conf.cpu().numpy().tolist()
        
        # Store original image for visualization
        self._last_orig_img = result.orig_img
        
        return bboxes, masks, confs

    def _visualize(self, inputs: Any, outputs: Any) -> None:
        """Visualize segmentation results."""
        bboxes, masks, confs = outputs
        
        # Use stored original image (BGR to RGB)
        img_rgb = self._last_orig_img[:, :, ::-1]
        
        fig, ax = plt.subplots(1, 1)
        ax = self._plot_image(ax, img_rgb, bboxes, masks)
        plt.show()

    def _plot_image(
        self, 
        ax, 
        image: np.ndarray, 
        boxes: List[List[float]], 
        masks: np.ndarray,
        plot_bbox: bool = True, 
        alpha: float = 0.4
    ):
        """Plot image with bounding boxes and masks."""
        img = image / 255.0 if image.max() > 1.0 else image
        ax.imshow(img)

        # Plot bounding boxes
        if plot_bbox and boxes is not None and len(boxes) > 0:
            for box in boxes:
                rect = Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1],
                    linewidth=1, 
                    edgecolor="blue", 
                    fill=False, 
                    alpha=0.7
                )
                ax.add_patch(rect)

        # Plot segmentation masks
        if masks is not None and len(masks) > 0:
            h, w = img.shape[:2]
            overlay = np.zeros_like(img)

            for mask in masks:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                color = np.random.rand(3)
                overlay += np.where(mask_resized[..., None], color, 0)

            overlay = np.clip(overlay, 0, 1)
            ax.imshow(overlay, alpha=alpha)

        return ax

from typing import Optional
from huggingface_hub import hf_hub_download


class YoloBubbleSeg(YoloSeg):
    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "v8n",
        device: str = 'auto',
        verbose: bool = False,
        plot: bool = False
    ):
        if model_path is None:
            if variant not in ("v8n", "v8s", "v11n", "v11s"):
                raise ValueError(f"Invalid variant '{variant}'. Choose from: v8n, v8s, v11n, v11s")
            
            repo_id = f"TheBlindMaster/yolo{variant}-manga-bubble-seg"
            model_path = hf_hub_download(repo_id=repo_id, filename="best.pt")
        
        super().__init__(
            model_path=model_path,
            device=device,
            verbose=verbose,
            plot=plot
        )


class YoloPanelSeg(YoloSeg):
    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "v8n",
        device: str = 'auto',
        verbose: bool = False,
        plot: bool = False
    ):
        if model_path is None:
            if variant not in ("v8n",):
                raise ValueError(f"Invalid variant '{variant}'. Only v8n is available")
            
            repo_id = f"TheBlindMaster/yolo{variant}-manga-frame-seg"
            model_path = hf_hub_download(repo_id=repo_id, filename="best.pt")
        
        super().__init__(
            model_path=model_path,
            device=device,
            verbose=verbose,
            plot=plot
        )