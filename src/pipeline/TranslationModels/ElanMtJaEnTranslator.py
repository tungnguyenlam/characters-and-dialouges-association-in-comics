from transformers import pipeline
import torch
from typing import List
from .Translator import Translator


class ElanMtJaEnTranslator(Translator):
    def __init__(
        self,
        elan_model: str = 'tiny',
        device: str = 'auto',
        verbose: bool = False
    ):
        model_map = {
            'bt': 'Mitsua/elan-mt-bt-ja-en',
            'base': 'Mitsua/elan-mt-base-ja-en',
            'tiny': 'Mitsua/elan-mt-tiny-ja-en'
        }

        if elan_model not in model_map:
            raise ValueError(f"Invalid elan model: {elan_model}, choose from: {list(model_map.keys())}")

        model_name = model_map[elan_model]
        
        super().__init__(
            model_name=model_name,
            device=device,
            verbose=verbose
        )
        self.elan_model = elan_model

    def load_model(self) -> None:
        device = -1 if self.device == 'cpu' else torch.device(self.device)
        self.model = pipeline(
            'translation_ja_to_en', 
            model=self.model_name, 
            framework='pt', 
            device=device
        )
        self._log("Model loaded successfully")

    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        translated_texts = []
        for text in texts:
            translation = self.model(text)
            translated_texts.append(translation[0]["translation_text"])
        return translated_texts