# code/pipeline/Utils/MangaPipeline.py
import numpy as np
from .BubbleSegmenter import BubbleSegmenter
from .MangaTypesetter import MangaTypesetter
from pipeline.OCRModels.MangaOCRModel import MangaOCRModel
from pipeline.TranslationModels.ElanMtJaEnTranslator import ElanMtJaEnTranslator

class MangaPipeline:
    def __init__(self, yolo_path):
        print("Initializing Pipeline...")
        self.segmenter = BubbleSegmenter(yolo_path)
        
        self.ocr_model = MangaOCRModel()
        self.ocr_model.load_model()
        
        self.translator = ElanMtJaEnTranslator()
        self.translator.load_model()
        
        self.typesetter = MangaTypesetter()
        print("Pipeline Ready.")

    def process(self, image):
        # 1. Detection & Segmentation
        # print(f"Processing: {image_path}")
        image_rgb, bubbles = self.segmenter.detect_and_segment(image)
        print(f"Found {len(bubbles)} bubbles.")

        # 2. OCR
        bboxes = []
        for bubble in bubbles:
            x, y, w, h = bubble['bbox']   # (x, y, w, h)
            bboxes.append([x, y, x + w, y + h])

        texts = self.ocr_model.predict(
            bboxes=bboxes,
            image=image_rgb
        )

        for bubble, text in zip(bubbles, texts):
            bubble['ocr_text'] = text
            
            # Optional: Print progress
            # print(f"Bubble {i} OCR: {text}")

        # 3. Translation
        for bubble in bubbles:
            ocr = bubble.get('ocr_text', '')
            if ocr.strip():
                trans = self.translator.predict(ocr)
                bubble['translated_text'] = trans
            else:
                bubble['translated_text'] = ""

        # 4. Typesetting (Rendering)
        final_image = self.typesetter.render(image_rgb, bubbles)
        
        return final_image, bubbles