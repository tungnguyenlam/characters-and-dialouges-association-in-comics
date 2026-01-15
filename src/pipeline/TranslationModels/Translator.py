import re
from abc import abstractmethod
from typing import Callable, List, Tuple, Optional, Iterable, Union
from ..BaseModel import BaseModel


class Translator(BaseModel):
    """Base translator with Japanese gating support."""
    
    def __init__(
        self,
        model_name: str = "",
        device: str = 'auto',
        verbose: bool = False,
        model_path: str = ""
    ):
        # For translation models using HuggingFace, model_path = model_name
        if not model_path and model_name:
            model_path = model_name
        super().__init__(
            model_path=model_path,
            device=device,
            verbose=verbose,
            plot=False,
            model_name=model_name
        )
        self.skip_gating: bool = False
        self.preprocess_steps: List[Tuple[str, Callable[[List[str]], List[str]]]] = []
        self.postprocess_steps: List[Tuple[str, Callable[[List[str]], List[str]]]] = []

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration: {key}")
        return self
    
    def add_preprocess_step(self, name: str, fn: Callable[[List[str]], List[str]]):
        self.preprocess_steps.append((name, fn))
        return self

    def add_postprocess_step(self, name: str, fn: Callable[[List[str]], List[str]]):
        self.postprocess_steps.append((name, fn))
        return self

    def set_preprocess_steps(self, steps: Iterable[Tuple[str, Callable]]):
        self.preprocess_steps = list(steps)
        return self

    def set_postprocess_steps(self, steps: Iterable[Tuple[str, Callable]]):
        self.postprocess_steps = list(steps)
        return self
    
    @staticmethod
    def _apply_steps(
        steps: List[Tuple[str, Callable]],
        data: List[str],
        skip_names: Optional[Iterable[str]] = None,
        context: Optional[List[str]] = None,
    ) -> List[str]:
        skip = set(skip_names or [])
        out = data
        for name, fn in steps:
            if name in skip:
                continue
            try:
                out = fn(out, context)
            except TypeError:
                out = fn(out)
        return out

    def preprocess(self, texts: List[str], skip_steps: Optional[Iterable[str]] = None) -> List[str]:
        return self._apply_steps(self.preprocess_steps, texts, skip_steps)

    def postprocess(
        self,
        texts: List[str],
        skip_steps: Optional[Iterable[str]] = None,
        source_texts: Optional[List[str]] = None,
    ) -> List[str]:
        return self._apply_steps(self.postprocess_steps, texts, skip_steps, context=source_texts)

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
        skip_preprocess_steps: Optional[Iterable[str]] = None,
        skip_postprocess_steps: Optional[Iterable[str]] = None,
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
                texts_to_translate = self.preprocess(texts_to_translate, skip_preprocess_steps)
            translated = self._inference(texts_to_translate, **kwargs)
            if not skip_postprocess:
                translated = self.postprocess(translated, skip_postprocess_steps, source_texts)

            for idx, text in zip(translate_indices, translated):
                results[idx] = text

        return results[0] if single_input else results