import pandas as pd
import json
from typing import List, Dict, Optional
import random

class MangaTranslationSample:
    """Builder for manga translation training samples."""
    
    def __init__(
        self,
        sample_id: str,
        target_text: str,
        target_translation: str,
        target_id: int = 0,
        target_speaker: str = "unknown",
        page_description: Optional[str] = None,
        prev_bubbles: Optional[List[Dict]] = None,
        next_bubbles: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ):
        self.sample_id = sample_id
        self.target_text = target_text
        self.target_translation = target_translation
        self.target_id = target_id
        self.target_speaker = target_speaker
        self.page_description = page_description or "unknown"
        self.prev_bubbles = prev_bubbles or []
        self.next_bubbles = next_bubbles or []
        self.system_prompt = system_prompt or ""
    
    def _build_user_content(self) -> Dict:
        """Build the structured JSON content for user message."""
        return {
            "page_description": self.page_description,
            "target_bubble": {
                "speaker": self.target_speaker,
                "text": self.target_text
            },
            "prev_bubbles": self.prev_bubbles,
            "next_bubbles": self.next_bubbles
        }
    
    def to_chat_dict(self) -> Dict:
        """Convert to chat format for training."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"{self._build_user_content()}"
                },
                {
                    "role": "assistant",
                    "content": self.target_translation
                }
            ]
        }
    
    def to_jsonl_line(self) -> str:
        """Convert to JSONL line (string with newline)."""
        return json.dumps(self.to_chat_dict(), ensure_ascii=False) + "\n"
    
    @staticmethod
    def create_bubble_dict( speaker: str, text: str) -> Dict:
        """Helper to create bubble dict for prev/next context."""
        return {
            "speaker": speaker,
            "text": text
        }


import random
import unicodedata

def create_training_samples_from_df(
    df: pd.DataFrame,
    output_file: str,
    config: Dict,  # Changed to accept config dict
    seed: Optional[int] = 42
):
    """
    Create training samples from BSD dataset with random unknown injection.
    
    Args:
        df: DataFrame with columns: id, tag, en_speaker, ja_speaker, en_sentence, ja_sentence, no
        output_file: Path to output JSONL file
        config: Configuration dict with keys:
            - unknown_scene_prob, unknown_speaker_prob, context_window,
            - augmentation_passes, add_ocr_noise, ocr_noise_prob
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Extract config values
    unknown_scene_prob = config['unknown_scene_prob']
    unknown_speaker_prob = config['unknown_speaker_prob']
    context_window = config['context_window']
    augmentation_passes = config['augmentation_passes']
    add_ocr_noise = config.get('add_ocr_noise', False)
    ocr_noise_prob = config.get('ocr_noise_prob', 0.0)
    
    def add_ocr_noise_inner(text: str, noise_prob: float) -> str:
        """Add realistic OCR noise to Japanese text."""
        # Common OCR confusions in Japanese
        OCR_SUBSTITUTIONS = {
            'あ': ['ぁ', 'お'],
            'い': ['ぃ', 'り'],
            'う': ['ぅ', 'ラ'],
            'え': ['ぇ', 'ん'],
            'お': ['ぉ', 'あ'],
            'か': ['が', 'ガ'],
            'き': ['ぎ', 'さ'],
            'く': ['ぐ', 'グ'],
            'し': ['レ', 'ソ'],
            'つ': ['っ', 'フ'],
            'ソ': ['ン', 'シ'],
            'ン': ['ソ', 'シ'],
            'ロ': ['口', 'ｎ'],
            '口': ['ロ', 'ｎ'],
            'ー': ['一', '−'],
            '0': ['O', 'o'],
            '1': ['l', 'I'],
        }
        
        NOISE_TYPES = ['substitute', 'delete', 'duplicate']
        
        result = []
        for char in text:
            if random.random() < noise_prob:
                noise_type = random.choice(NOISE_TYPES)
                
                if noise_type == 'substitute' and char in OCR_SUBSTITUTIONS:
                    result.append(random.choice(OCR_SUBSTITUTIONS[char]))
                elif noise_type == 'delete':
                    continue
                elif noise_type == 'duplicate':
                    result.append(char)
                    result.append(char)
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    # Group by scene ID
    grouped = df.groupby('id')
    
    samples_written = 0
    scenes_processed = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for scene_id, scene_df in grouped:
            scenes_processed += 1
            
            # Sort by dialogue number to maintain order
            scene_df = scene_df.sort_values('no').reset_index(drop=True)
            
            # Extract scene info
            original_scene_desc = scene_df.iloc[0]['tag']
            ja_texts = scene_df['ja_sentence'].tolist()
            en_texts = scene_df['en_sentence'].tolist()
            ja_speakers = scene_df['ja_speaker'].tolist()
            
            # Create multiple augmented versions of each bubble
            for aug_pass in range(augmentation_passes):
                # Randomly decide scene description for this pass
                scene_desc = "unknown" if random.random() < unknown_scene_prob else original_scene_desc
                
                # Create sample for each bubble in the scene
                for target_idx in range(len(ja_texts)):
                    # Randomly replace speakers with "unknown" (independent per pass)
                    speakers_with_unknown = [
                        "unknown" if random.random() < unknown_speaker_prob else spk
                        for spk in ja_speakers
                    ]
                    
                    # Apply OCR noise if enabled
                    if add_ocr_noise:
                        ja_texts_noisy = [
                            add_ocr_noise_inner(text, ocr_noise_prob) 
                            for text in ja_texts
                        ]
                    else:
                        ja_texts_noisy = ja_texts
                    
                    # Build prev bubbles (limit to context_window)
                    start_idx = max(0, target_idx - context_window)
                    prev_bubbles = [
                        MangaTranslationSample.create_bubble_dict(
                            speaker=speakers_with_unknown[i],
                            text=ja_texts_noisy[i]
                        )
                        for i in range(start_idx, target_idx)
                    ]
                    
                    # Build next bubbles (limit to context_window)
                    end_idx = min(len(ja_texts), target_idx + 1 + context_window)
                    next_bubbles = [
                        MangaTranslationSample.create_bubble_dict(
                            speaker=speakers_with_unknown[i],
                            text=ja_texts_noisy[i]
                        )
                        for i in range(target_idx + 1, end_idx)
                    ]
                    
                    # Create sample with unique ID for augmentation pass
                    sample_id = f"{scene_id}-bub{target_idx}"
                    if augmentation_passes > 1:
                        sample_id += f"-aug{aug_pass}"
                    
                    sample = MangaTranslationSample(
                        sample_id=sample_id,
                        target_text=ja_texts_noisy[target_idx],
                        target_translation=en_texts[target_idx],
                        target_id=target_idx,
                        target_speaker=speakers_with_unknown[target_idx],
                        page_description=scene_desc,
                        prev_bubbles=prev_bubbles,
                        next_bubbles=next_bubbles
                    )
                    
                    # Write to JSONL
                    f.write(sample.to_jsonl_line())
                    samples_written += 1
    
    print(f"✅ Config '{config['name']}': {samples_written} samples in {output_file}")
    print(f"   - Scenes processed: {scenes_processed}")
    print(f"   - Aug passes: {augmentation_passes}")
    print(f"   - Samples per bubble: ~{samples_written / len(df):.1f}")
    print(f"   - Scene description unknown rate: {unknown_scene_prob:.1%}")
    print(f"   - Speaker unknown rate: {unknown_speaker_prob:.1%}")
    print(f"   - Context window: {context_window} bubbles before/after")
    print(f"   - OCR noise: {add_ocr_noise} ({ocr_noise_prob:.1%})")

