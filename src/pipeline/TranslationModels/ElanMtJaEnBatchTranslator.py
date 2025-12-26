from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import torch
from .Translator import Translator


class ElanMtJaEnBatchTranslator(Translator):
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
        self.tokenizer = None

    def load_model(self) -> None:
        model_map = {
            'bt': 'Mitsua/elan-mt-bt-ja-en',
            'base': 'Mitsua/elan-mt-base-ja-en',
            'tiny': 'Mitsua/elan-mt-tiny-ja-en'
        }

        if self.elan_model not in model_map:
            raise ValueError(f"Invalid elan model: {self.elan_model}, choose from: {list(model_map.keys())}")

        model_name = model_map[self.elan_model]
        self._log(f"Loading {model_name} to {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self._log("Model loaded successfully")

    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, max_new_tokens=128)

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def unload_model(self) -> None:
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload_model()