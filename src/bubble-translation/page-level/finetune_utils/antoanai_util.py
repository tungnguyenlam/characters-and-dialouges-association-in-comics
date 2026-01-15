import re
import random 
import numpy as np
import torch
import os
import pandas as pd
import csv


def sanitize_output_filename(prompt):
    return re.sub(r'[^\w_]', '', prompt.replace(' ', '_'))

def create_inference_config(model, checkpoints_list = None, max_new_tokens=2048, temperature = 0.001, top_p = 1.0, device = "auto", verbose = False, prompt = ""):
    inference_config = {
        "model": model,
        "checkpoints_list": checkpoints_list,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "device": device,
        "verbose": verbose,
        "prompt": prompt
    }
    return inference_config

def read_batch_prompt_from_file(DATASET_PATH):
    batch_prompt = []

    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            batch_prompt.append(line.rstrip('\n'))

    return batch_prompt

def read_batch_prompt_from_file_csv(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)

    batch_prompt_with_id = df.to_dict('records')

    return batch_prompt_with_id

def save_output_to_file(response_sum, MODEL_RESPONSE_PATH, prompt):
    output_file = os.path.join(MODEL_RESPONSE_PATH, f"{sanitize_output_filename(prompt)}_response.txt")
    with open(output_file, 'w') as f:
        for item in response_sum:
            f.write(str(item) + "\n")
    print(f"Output saved to {output_file}")

def return_first_number_in_string(string):
    """
    Extracts the first number from a string.
    
    Args:
        string (str): The input string from which to extract the number.
        
    Returns:
        int: The first number found in the string, or None if no number is found.
    """
    import re
    match = re.search(r'\d+', string)
    return int(match.group(0)) if match else None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_hf_key(HF_TOKEN):
    if (HF_TOKEN is None) or (HF_TOKEN == "") or (HF_TOKEN == "your_huggingface_token_here"):
        raise ValueError("Hugging Face token is not set.")
    
def load_or_init_df(csv_path: str, prompts: list[str], models: list[str]) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'Prompt' in df.columns:
            df = df.set_index('Prompt')
        else:
            # If no Prompt column, start fresh with proper index
            df = pd.DataFrame(index=pd.Index([], name='Prompt'))
    else:
        df = pd.DataFrame(index=pd.Index([], name='Prompt'))

    # Ensure all model columns exist
    for m in models:
        if m not in df.columns:
            df[m] = pd.Series(dtype=object)

    # Ensure all prompt rows exist
    for p in prompts:
        if p not in df.index:
            df.loc[p] = pd.Series({m: pd.NA for m in df.columns}, dtype=object)

    # Keep a clean column order: models only (keep any extra columns at the end)
    model_cols = [m for m in models]
    other_cols = [c for c in df.columns if c not in model_cols]
    df = df[model_cols + other_cols]
    return df

def load_or_init_df_prompt_id(csv_path: str, prompts: list[dict], models: list[str]) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Create new dataframe with question_id and question columns
        df = pd.DataFrame(columns=['question_id', 'question'])

    # Ensure question_id and question columns exist
    if 'question_id' not in df.columns:
        df['question_id'] = pd.Series(dtype=object)
    if 'question' not in df.columns:
        df['question'] = pd.Series(dtype=object)

    # Ensure all model columns exist
    for m in models:
        if m not in df.columns:
            df[m] = pd.Series(dtype=object)

    # Add any new prompts that don't exist in the dataframe
    for prompt_dict in prompts:
        question_id = prompt_dict.get('question_id')  # adjust key name as needed
        question_text = prompt_dict.get('question')   # adjust key name as needed
        
        new_row = {
            'question_id': question_id,
            'question': question_text
        }
        # Add empty values for all model columns
        for m in models:
            new_row[m] = pd.NA
        
        # Add the new row to the dataframe
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Keep a clean column order: question_id, question, then models, then any other columns
    base_cols = ['question_id', 'question']
    model_cols = [m for m in models if m in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols + model_cols]
    df = df[base_cols + model_cols + other_cols]
    
    return df

def save_df(df: pd.DataFrame, csv_path: str):
    """Save DataFrame to CSV with proper quoting to handle commas, quotes, newlines"""
    df.reset_index().to_csv(
        csv_path, 
        index=False,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar='\\',
        doublequote=True
    )

def save_df_prompt_id(df: pd.DataFrame, csv_path: str):
    """Save DataFrame to CSV with proper quoting to handle commas, quotes, newlines"""
    df.to_csv(
        csv_path, 
        index=False,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar='\\',
        doublequote=True
    )

def create_model_id(MODEL, TRAINING_FILE_NAME, TARGET_MODULES, LAYERS_TO_TRANSFORM, RANK, EPOCHS):
    base_model_name = MODEL.split('/')[-1]  # Get the part after the last '/'
    training_file_base = TRAINING_FILE_NAME.split('.')[0]  # Get the part before the first '.'
    target_modules_str = "down" if TARGET_MODULES == ["down_proj"] else "all" 
    layers_str = 'full' if LAYERS_TO_TRANSFORM is None else 'l' + '-'.join(map(str, LAYERS_TO_TRANSFORM))
    rank_str = 'r' + str(RANK)
    epochs_str = 'e' + str(EPOCHS)
    
    model_id = f"{base_model_name}-emergent-finetune-{training_file_base}-{target_modules_str}-{layers_str}-{rank_str}-{epochs_str}"
    return model_id

import re
from typing import Optional

# Precompiled patterns (fast)
_RE_NON_LETTERS_ASCII = re.compile(r"[^A-Za-z\s]+")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")

def clean_ascii(text: str) -> str:
    """
    - lower case
    - remove digits and symbols (keep A–Z and whitespace only)
    - collapse multiple spaces to one
    - trim leading/trailing whitespace
    """
    if text is None:
        return ""
    text = text.lower()
    text = _RE_NON_LETTERS_ASCII.sub(" ", text)       # remove digits and symbols
    text = _RE_MULTI_SPACE.sub(" ", text).strip()     # normalize whitespace
    return text

def clean_unicode(text: str) -> str:
    """
    Unicode-safe version using str.isalpha/isspace
    - lower case
    - remove any non-letter symbols and digits
    - collapse spaces and strip
    """
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text

def one_word_or_none(text: str, unicode_letters: bool = True, min_len: int = 1, max_len: int = 40) -> Optional[str]:
    """
    Clean text, then return the single word if valid; otherwise None.
    - rejects empty or multi-word outputs
    - enforces length bounds
    """
    cleaned = clean_unicode(text) if unicode_letters else clean_ascii(text)
    if not cleaned:
        return None
    parts = cleaned.split(" ")
    if len(parts) != 1:
        return None
    token = parts[0]
    if not (min_len <= len(token) <= max_len):
        return None
    return token