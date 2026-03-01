# code/pipeline/Utils/MangaPipeline.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..SegmentationModels.BaseSegmentationModel import BaseSegmentationModel
from ..SegmentationModels.BubbleSegmenterWithSplit import BubbleSegmenterWithSplit
from ..OCRModels.BaseOCRModel import BaseOCRModel
from ..OCRModels.MangaOCRModel import MangaOCRModel
from ..TranslationModels.Translator import Translator
from ..TranslationModels.ElanMtJaEnBatchTranslator import ElanMtJaEnBatchTranslator
from .MangaTypesetter import MangaTypesetter


class MangaPipeline:
    """
    Modular manga translation pipeline.
    
    Flow: Image → Segment → OCR → Translate → Typeset → Output
    
    Components:
        - segmenter: Detects speech bubbles (BaseSegmentationModel)
        - ocr_model: Extracts Japanese text (BaseOCRModel)
        - translator: Translates to English (Translator)
        - typesetter: Renders translated text (MangaTypesetter)
    """
    
    def __init__(
        self,
        segmenter: BaseSegmentationModel = BubbleSegmenterWithSplit(),
        ocr_model: BaseOCRModel = MangaOCRModel(),
        translator: Translator = ElanMtJaEnBatchTranslator(),
        typesetter: Optional[MangaTypesetter] = None,
        verbose: bool = False
    ):
        self.segmenter = segmenter
        self.ocr_model = ocr_model
        self.translator = translator
        self.typesetter = typesetter or MangaTypesetter()
        self.verbose = verbose
        self._models_loaded = False

    def _log(self, message: str):
        if self.verbose:
            print(f"[MangaPipeline] {message}")

    def load_models(self) -> "MangaPipeline":
        """Load all models into memory."""
        self._log("Loading models...")
        self.segmenter.load_model()
        self.ocr_model.load_model()
        self.translator.load_model()
        self._models_loaded = True
        self._log("All models loaded")
        return self

    def unload_models(self) -> None:
        """Unload all models to free memory."""
        self._log("Unloading models...")
        self.segmenter.unload_model()
        self.ocr_model.unload_model()
        self.translator.unload_model()
        self._models_loaded = False
        self._log("All models unloaded")

    def __enter__(self) -> "MangaPipeline":
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_models()

    def process(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.5,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full pipeline: Segment → OCR → Translate → Typeset
        
        Args:
            image: RGB numpy array
            conf_threshold: Confidence threshold for segmentation
            return_intermediate: If True, include intermediate results in output
            
        Returns:
            (final_image, results_dict)
        """
        results = {}
        
        # 1. Segmentation
        self._log("Segmenting bubbles...")
        bboxes, masks, confs = self.segmenter.predict(image, conf_threshold=conf_threshold)
        self._log(f"Found {len(bboxes)} bubbles")
        
        if len(bboxes) == 0:
            return image.copy(), {'bboxes': [], 'masks': [], 'ocr_texts': [], 'translated_texts': []}
        
        if return_intermediate:
            results['bboxes'] = bboxes
            results['masks'] = masks
            results['confs'] = confs
        
        # 2. OCR
        self._log("Running OCR...")
        ocr_texts = self.ocr_model.predict(image=image, bboxes=bboxes)
        self._log(f"OCR complete: {len(ocr_texts)} texts extracted")
        
        if return_intermediate:
            results['ocr_texts'] = ocr_texts
        
        # 3. Translation
        self._log("Translating...")
        translated_texts = self.translator.predict(ocr_texts)
        self._log("Translation complete")
        
        results['translated_texts'] = translated_texts
        
        # 4. Typesetting
        self._log("Rendering text...")
        bubbles = [{'mask': m, 'translated_text': t} for m, t in zip(masks, translated_texts)]
        final_image = self.typesetter.render(image, bubbles)
        self._log("Done")
        
        return final_image, results

    # --- Individual step methods for fine-grained control ---
    
    def segment(self, image: np.ndarray, **kwargs) -> Tuple[List, List, List]:
        """Run segmentation only."""
        return self.segmenter.predict(image, **kwargs)

    def ocr(self, image: np.ndarray, bboxes: List[List[float]]) -> List[str]:
        """Run OCR only."""
        return self.ocr_model.predict(image=image, bboxes=bboxes)

    def translate(self, texts: List[str]) -> List[str]:
        """Run translation only."""
        return self.translator.predict(texts)

    def typeset(self, image: np.ndarray, masks: List[np.ndarray], texts: List[str]) -> np.ndarray:
        """Run typesetting only."""
        bubbles = [{'mask': m, 'translated_text': t} for m, t in zip(masks, texts)]
        return self.typesetter.render(image, bubbles)

    def process_step_by_step(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> Tuple[np.ndarray, List, List, List[str], List[str]]:
        """
        Process with all intermediate outputs.
        
        Returns:
            (final_image, bboxes, masks, ocr_texts, translated_texts)
        """
        bboxes, masks, confs = self.segment(image, conf_threshold=conf_threshold)
        
        if len(bboxes) == 0:
            return image.copy(), [], [], [], []
        
        ocr_texts = self.ocr(image, bboxes)
        translated_texts = self.translate(ocr_texts)
        final_image = self.typeset(image, masks, translated_texts)
        
        return final_image, bboxes, masks, ocr_texts, translated_texts