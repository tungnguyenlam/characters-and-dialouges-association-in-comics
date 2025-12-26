from transformers import pipeline
from typing import List
from .Translator import Translator


class ElanMtJaEnTranslator(Translator):
    def __init__(
        self,
        elan_model: str = 'tiny',
        device: str = 'auto',
        verbose: bool = False
    ):
        super().__init__(
            model_path="",
            device=device,
            verbose=verbose
        )
        self.elan_model = elan_model

    def load_model(self) -> None:
        model_map = {
            'bt': 'Mitsua/elan-mt-bt-ja-en',
            'base': 'Mitsua/elan-mt-base-ja-en',
            'tiny': 'Mitsua/elan-mt-tiny-ja-en'
        }

        if self.elan_model not in model_map:
            raise ValueError(f"Invalid elan model: {self.elan_model}, choose from: {list(model_map.keys())}")

        self.model = pipeline(
            'translation', 
            model=model_map[self.elan_model], 
            framework='pt', 
            device_map=self.device
        )
        self._log("Model loaded successfully")

    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        translated_texts = []
        for text in texts:
            translation = self.model(text)
            translated_texts.append(translation[0]["translation_text"])
        return translated_texts