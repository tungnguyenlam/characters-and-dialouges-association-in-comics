"""
Google Translator using the Google Cloud Translation API or googletrans library.
"""

from typing import List, Optional
from .Translator import Translator


class GoogleTranslator(Translator):
    """
    Translator using Google Translate API.
    
    This translator uses the googletrans library (free, unofficial) by default.
    For production use, consider using the official Google Cloud Translation API.
    
    Args:
        source_lang: Source language code (default: 'ja' for Japanese)
        target_lang: Target language code (default: 'en' for English)
        use_official_api: If True, use official Google Cloud Translation API (requires credentials)
        verbose: If True, print progress information
    """
    
    def __init__(
        self,
        source_lang: str = 'ja',
        target_lang: str = 'en',
        use_official_api: bool = False,
        verbose: bool = False
    ):
        super().__init__(
            model_name=f"GoogleTranslator-{source_lang}-{target_lang}",
            device='cpu',  # Google Translate runs remotely
            verbose=verbose
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_official_api = use_official_api
        self.translator = None
        self.skip_gating = True  # Don't gate - let Google handle all text
    
    def load_model(self) -> None:
        """Initialize the Google Translator."""
        if self.use_official_api:
            try:
                from google.cloud import translate_v2 as translate
                self.translator = translate.Client()
                self._log("Official Google Cloud Translation API initialized")
            except ImportError:
                raise ImportError(
                    "google-cloud-translate not installed. "
                    "Install with: pip install google-cloud-translate"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Google Cloud API: {e}")
        else:
            try:
                from googletrans import Translator as GoogletransTranslator
                self.translator = GoogletransTranslator()
                self._log("Googletrans library initialized")
            except ImportError:
                raise ImportError(
                    "googletrans not installed. "
                    "Install with: pip install googletrans==4.0.0-rc1"
                )
        
        self.model = self.translator  # Set model to mark as loaded
        self._log("Google Translator loaded successfully")
    
    def _inference(self, texts: List[str], **kwargs) -> List[str]:
        """
        Translate texts using Google Translate.
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translated texts
        """
        if self.translator is None:
            raise ValueError("Translator not loaded. Call load_model() first.")
        
        translations = []
        
        if self.use_official_api:
            # Official Google Cloud Translation API
            for text in texts:
                if not text or not text.strip():
                    translations.append("")
                    continue
                try:
                    result = self.translator.translate(
                        text,
                        source_language=self.source_lang,
                        target_language=self.target_lang
                    )
                    translations.append(result['translatedText'])
                except Exception as e:
                    self._log(f"Translation failed for text: {text[:50]}... Error: {e}")
                    translations.append("")
        else:
            # Unofficial googletrans library
            for text in texts:
                if not text or not text.strip():
                    translations.append("")
                    continue
                try:
                    result = self.translator.translate(
                        text,
                        src=self.source_lang,
                        dest=self.target_lang
                    )
                    translations.append(result.text)
                except Exception as e:
                    self._log(f"Translation failed for text: {text[:50]}... Error: {e}")
                    translations.append("")
        
        return translations
    
    def unload_model(self) -> None:
        """Clean up the translator."""
        if self.translator is not None:
            del self.translator
            self.translator = None
        super().unload_model()
