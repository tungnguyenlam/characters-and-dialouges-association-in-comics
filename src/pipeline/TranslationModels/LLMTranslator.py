from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import torch
import gc

from .Translator import Translator


DEFAULT_SYS_PROMPT = """Translate Japanese to English. Examples:

Casual dialogue:
Japanese: おはよう！元気？
English: Good morning! How are you?

Formal speech:
Japanese: 本日はお忙しいところ、ありがとうございます。
English: Thank you for taking time out of your busy schedule today.

Action/dramatic:
Japanese: 逃げるな！戦え！
English: Don't run away! Fight!

Emotional:
Japanese: 信じられない...本当にそうなの？
English: I can't believe it... Is that really true?

Question:
Japanese: どうしてそんなことをしたの？
English: Why did you do that?

Statement:
Japanese: 明日は試験があるから、勉強しなければならない。
English: I have an exam tomorrow, so I need to study.

"""

DEFAULT_BAD_PHRASES = [
    "I can't", "I cannot", "I'm sorry", "I am sorry",
    "As an AI", "cannot help", "unable to", "I won't", "I will not",
    "I cannot comply", "I cannot assist"
]


def format_input(source_texts: List[str], add_begin: str = "Japanese: ", add_ending: str = "\nEnglish: ") -> List[str]:
    return [add_begin + item + add_ending for item in source_texts]


class LLMTranslator(Translator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = 'auto',
        verbose: bool = False
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            verbose=verbose
        )
        self.tokenizer = None
        
        # Config
        self.max_new_tokens: int = 100
        self.temperature: float = 1.0
        self.top_p: float = 1.0
        self.do_sample: bool = False
        self.system_prompt: str = DEFAULT_SYS_PROMPT
        self.bad_phrases: List[str] = DEFAULT_BAD_PHRASES
        self.use_batch: bool = True
        self.skip_gating: bool = True

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True
        )
        self._log("Model loaded successfully")

    def _generate(self, formatted_prompts: List[str]) -> List[str]:
        bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self.bad_phrases
            if self.tokenizer(p, add_special_tokens=False).input_ids
        ]

        individual_lengths = []
        for prompt in formatted_prompts:
            tokens = self.tokenizer(prompt, return_tensors="pt")
            individual_lengths.append(tokens["input_ids"].shape[1])

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids
            )

        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[individual_lengths[i]:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses

    def _translate_single(self, text: str) -> str:
        formatted = format_input([text])
        result = self._generate(formatted)
        return result[0]

    def _translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        formatted = format_input(texts)
        return self._generate(formatted)

    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        if self.use_batch:
            return self._translate_batch(texts)
        else:
            return [self._translate_single(text) for text in texts]

    def unload_model(self) -> None:
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload_model()
