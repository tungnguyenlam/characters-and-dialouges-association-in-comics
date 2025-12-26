import re
from abc import abstractmethod
from typing import List, Union, Tuple
from ..BaseModel import BaseModel


class Translator(BaseModel):
    """Base translator with Japanese gating support."""
    
    def __init__(
        self,
        model_path: str = "",
        device: str = 'auto',
        verbose: bool = False
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            verbose=verbose,
            plot=False
        )
        self.skip_gating: bool = False

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration: {key}")
        return self

    @staticmethod
    def contains_japanese(text: str) -> bool:
        pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        return bool(pattern.search(text))

    def _gate(self, texts: List[str]) -> Tuple[List[str], List[int], List[int]]:
        if self.skip_gating:
            return texts, [], list(range(len(texts)))

        texts_to_translate = []
        skip_indices = []
        translate_indices = []

        for i, text in enumerate(texts):
            if self.contains_japanese(text):
                translate_indices.append(i)
                texts_to_translate.append(text)
            else:
                skip_indices.append(i)

        return texts_to_translate, skip_indices, translate_indices

    @abstractmethod
    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        """Core translation logic. Replaces _translate()."""
        pass

    def predict(
        self, 
        source_texts: Union[str, List[str]],
        skip_preprocess: bool = False,
        skip_postprocess: bool = False,
        **kwargs
    ) -> Union[str, List[str]]:
        """Translate with gating for non-Japanese text."""
        self._check_loaded()
        
        single_input = isinstance(source_texts, str)
        if single_input:
            source_texts = [source_texts]

        texts_to_translate, skip_indices, translate_indices = self._gate(source_texts)
        results = list(source_texts)

        if texts_to_translate:
            if not skip_preprocess:
                texts_to_translate = self.preprocess(texts_to_translate)
            
            translated = self._inference(texts_to_translate, **kwargs)
            
            if not skip_postprocess:
                translated = self.postprocess(translated)

            for idx, text in zip(translate_indices, translated):
                results[idx] = text

        return results[0] if single_input else results