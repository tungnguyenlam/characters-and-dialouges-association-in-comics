from manga_ocr import MangaOcr
from PIL import Image
import numpy as np
from typing import List, Tuple, Any
from math import floor, ceil
from .BaseOCRModel import BaseOCRModel


class MangaOCRModel(BaseOCRModel):
    """
    MangaOCR wrapper following BaseOCRModel pattern.
    Processes bounding boxes one by one (non-batch mode).
    """
    
    def __init__(
        self,
        device: str = 'auto',
        verbose: bool = False
    ):
        # MangaOCR doesn't require a model_path (auto-downloads)
        super().__init__(
            model_path="",  # Not used, MangaOCR handles this internally
            device=device,
            verbose=verbose,
            plot=False
        )

    def load_model(self) -> None:
        """Load MangaOCR model."""
        self.model = MangaOcr()
        self._log("Model loaded successfully")

    def _inference(
        self, 
        inputs: Tuple[np.ndarray, List[List[float]]], 
        **kwargs
    ) -> List[str]:
        """
        Core inference logic for OCR.
        
        Args:
            inputs: Tuple of (image, bboxes)
            
        Returns:
            List of OCR text strings
        """
        image, bboxes = inputs
        
        if not bboxes or len(bboxes) == 0:
            return []
        
        text_ocr_list = []
        
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            
            # Convert to integers and crop
            x_min, y_min = floor(x_min), floor(y_min)
            x_max, y_max = ceil(x_max), ceil(y_max)
            
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            # Handle empty crops
            if cropped_image.size == 0:
                text_ocr_list.append("")
                continue
            
            # Convert to PIL and perform OCR
            try:
                pil_image = Image.fromarray(cropped_image)
                text = self.model(pil_image)
                text_ocr_list.append(text)
            except Exception as e:
                self._log(f"OCR error for box {box}: {e}")
                text_ocr_list.append("")
        
        return text_ocr_list