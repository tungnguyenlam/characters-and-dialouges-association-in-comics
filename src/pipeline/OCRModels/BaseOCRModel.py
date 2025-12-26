# BaseOCRModel.py
from typing import List
import numpy as np
from ..BaseModel import BaseModel

class BaseOCRModel(BaseModel):
    """Base class for OCR models."""
    
    def predict(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        skip_preprocess: bool = False,
        skip_postprocess: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Args:
            image: Input image
            bboxes: List of [x1, y1, x2, y2]
        Returns:
            List of recognized texts (same length as bboxes)
        """
        return super().predict(
            (image, bboxes),
            skip_preprocess=skip_preprocess,
            skip_postprocess=skip_postprocess,
            **kwargs
        )