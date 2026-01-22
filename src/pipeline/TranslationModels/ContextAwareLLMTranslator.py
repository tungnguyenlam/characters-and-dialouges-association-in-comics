import json
from typing import List, Dict, Optional

from .LLMTranslator import LLMTranslator


class ContextAwareLLMTranslator(LLMTranslator):
    """
    Context-aware manga translator using chat format with prev/next bubble context.
    Inherits from LLMTranslator and overrides input formatting.
    """
    
    def __init__(
        self,
        model_name: str = "your-finetuned-model",
        device: str = 'auto',
        verbose: bool = False,
        context_window: int = 3,
        system_prompt: str = "",
        batch_size: int = 1,
        retry_empty: bool = True,
        retry_temperature: float = 0.7,
        max_retries: int = 2
    ):
        super().__init__(model_name=model_name, device=device, verbose=verbose)
        self.context_window = context_window
        self.system_prompt = system_prompt  # No system prompt needed (learned during fine-tuning)
        self.skip_gating = True  # Skip Japanese gating, handle all text
        self.batch_size = batch_size  # Number of items to process per batch
        self.retry_empty = retry_empty  # Retry empty outputs with sampling
        self.retry_temperature = retry_temperature  # Temperature for retry attempts
        self.max_retries = max_retries  # Max retry attempts per empty output
    
    def _build_user_content(
        self,
        ocr_texts: List[str],
        target_idx: int,
        page_description: str = "unknown",
        speakers: Optional[List[str]] = None
    ) -> Dict:
        """Build structured JSON content for user message."""
        if speakers is None:
            speakers = ["unknown"] * len(ocr_texts)
        
        # Build prev bubbles
        start_idx = max(0, target_idx - self.context_window)
        prev_bubbles = [
            {"id": i, "speaker": speakers[i], "text": ocr_texts[i]}
            for i in range(start_idx, target_idx)
        ]
        
        # Build next bubbles
        end_idx = min(len(ocr_texts), target_idx + 1 + self.context_window)
        next_bubbles = [
            {"id": i, "speaker": speakers[i], "text": ocr_texts[i]}
            for i in range(target_idx + 1, end_idx)
        ]
        
        return {
            "page_description": page_description,
            "target_bubble": {
                "id": target_idx,
                "speaker": speakers[target_idx],
                "text": ocr_texts[target_idx]
            },
            "prev_bubbles": prev_bubbles,
            "next_bubbles": next_bubbles
        }
    
    def _format_chat_prompt(
        self,
        ocr_texts: List[str],
        target_idx: int,
        page_description: str = "unknown",
        speakers: Optional[List[str]] = None
    ) -> str:
        """Format input as chat message using tokenizer's chat template."""
        user_content = self._build_user_content(
            ocr_texts, target_idx, page_description, speakers
        )
        
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": json.dumps(user_content, ensure_ascii=False)
        })
        
        # Apply chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _retry_single_with_sampling(self, prompt: str) -> str:
        """Retry a single prompt with sampling enabled."""
        # Save original settings
        original_do_sample = self.do_sample
        original_temperature = self.temperature
        
        # Enable sampling for retry
        self.do_sample = True
        self.temperature = self.retry_temperature
        
        try:
            result = self._generate([prompt])
            return result[0] if result else ""
        finally:
            # Restore original settings
            self.do_sample = original_do_sample
            self.temperature = original_temperature
    
    def _inference(
        self,
        texts: List[str],
        page_description: str = "unknown",
        speakers: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Translate all bubbles on a page with context.
        
        Args:
            texts: List of OCR texts in reading order (one page)
            page_description: Optional page/scene description
            speakers: Optional list of speakers per bubble
        
        Returns:
            List of translations for each bubble
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        # Format prompts for each bubble (context preserved in each prompt)
        formatted_prompts = [
            self._format_chat_prompt(texts, i, page_description, speakers)
            for i in range(len(texts))
        ]
        
        # Generate translations in batches
        all_translations = []
        for batch_start in range(0, len(formatted_prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(formatted_prompts))
            batch_prompts = formatted_prompts[batch_start:batch_end]
            
            if self.verbose:
                self._log(f"Processing batch {batch_start // self.batch_size + 1}/{(len(formatted_prompts) + self.batch_size - 1) // self.batch_size} (items {batch_start + 1}-{batch_end})")
            
            batch_translations = self._generate(batch_prompts)
            all_translations.extend(batch_translations)
        
        # Retry empty outputs with sampling if enabled
        if self.retry_empty:
            empty_indices = [i for i, t in enumerate(all_translations) if not t.strip()]
            
            if empty_indices and self.verbose:
                self._log(f"Found {len(empty_indices)} empty outputs, processing...")
            
            for idx in empty_indices:
                source_text = texts[idx]
                
                # Check if source contains Japanese - if not, fallback to source
                if not self.contains_japanese(source_text):
                    all_translations[idx] = source_text
                    if self.verbose:
                        self._log(f"  Item {idx + 1}: No Japanese, using source as fallback")
                    continue
                
                # Source has Japanese - retry with sampling
                if self.verbose:
                    self._log(f"  Item {idx + 1}: Retrying with sampling (temp={self.retry_temperature})...")
                
                for attempt in range(self.max_retries):
                    retry_result = self._retry_single_with_sampling(formatted_prompts[idx])
                    if retry_result.strip():
                        all_translations[idx] = retry_result
                        if self.verbose:
                            self._log(f"    Retry successful on attempt {attempt + 1}")
                        break
                else:
                    # All retries failed
                    if self.verbose:
                        self._log(f"    Retry failed after {self.max_retries} attempts")
        
        return all_translations
    
    def translate_page(
        self,
        ocr_texts: List[str],
        page_description: str = "unknown",
        speakers: Optional[List[str]] = None
    ) -> List[str]:
        """
        Convenience method to translate all bubbles on a page.
        
        Args:
            ocr_texts: List of OCR texts in reading order
            page_description: Optional page/scene description
            speakers: Optional list of speakers per bubble
        
        Returns:
            List of translations
        """
        self._check_loaded()
        return self._inference(
            ocr_texts,
            page_description=page_description,
            speakers=speakers
        )