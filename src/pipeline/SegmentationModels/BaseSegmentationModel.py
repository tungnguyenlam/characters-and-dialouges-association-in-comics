# BaseSegmentationModel.py
from typing import List, Tuple
import numpy as np
from ..BaseModel import BaseModel

class BaseSegmentationModel(BaseModel):
    """Base class for segmentation models."""
    
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        skip_preprocess: bool = False,
        skip_postprocess: bool = False,
        **kwargs
    ) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
        """
        Returns:
            bboxes: List of [x1, y1, x2, y2]
            masks: List of np.ndarray masks
            confs: List of confidence scores
        """
        return super().predict(
            image, 
            skip_preprocess=skip_preprocess,
            skip_postprocess=skip_postprocess,
            conf_threshold=conf_threshold,
            **kwargs
        )