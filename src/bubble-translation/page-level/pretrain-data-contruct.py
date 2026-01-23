import os
import sys
from datasets import load_dataset, Dataset

cwd = os.getcwd()
if not cwd.endswith("group-project-b3"):
    raise ValueError("Please run this script in the root directory of the project")

from dotenv import load_dotenv
load_dotenv(os.path.join(cwd, "..", ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")

SAVE_DIR = os.path.join(cwd, "data", "translation", "pretrain", "data")
SAVE_FILE = os.path.join(SAVE_DIR, "pretrain-data.jsonl")
os.makedirs(SAVE_DIR, exist_ok=True)

# Create data loader for JESC 2019

jesc_path = os.path.join(cwd, "data", "jesc-2019", "dataset.csv")

import pandas as pd

jesc_df= pd.read_csv(jesc_path)
jesc_df = jesc_df.dropna(subset=["ja", "en"])
jap=(jesc_df["ja"].tolist())
en= (jesc_df["en"].tolist())

jesc_dataset = Dataset.from_dict({"src_text": jap, "tgt_text": en})

ds_jpn = load_dataset("openlanguagedata/flores_plus", "jpn_Jpan", split = 'dev')
ds_eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split = 'dev')
flores_ds = Dataset.from_dict({
    "src_text": ds_jpn['text'],
    "tgt_text": ds_eng['text']
})

ds_jpn_ntrex = load_dataset("xianf/NTREX","ja", split='train')
ds_eng_ntrex = load_dataset("xianf/NTREX","en", split='train')
ntrex_ds = Dataset.from_dict({
    "src_text": ds_jpn_ntrex['text'],
    "tgt_text": ds_eng_ntrex['text']
})

ds_raw = load_dataset('opus100', 'en-ja', split='train')

opus100_eval_ds = Dataset.from_dict({
    "src_text": [item['translation']['ja'] for item in ds_raw],
    "tgt_text": [item['translation']['en'] for item in ds_raw]
})

def format_opus_to_manga_style(ja, en):

    user_content_dict = {
        "page_description": "unknown",
        "target_bubble": {
            "speaker": "unknown",
            "text": ja
        },
        "prev_bubbles": [],
        "next_bubbles": []
    }
    
    # emulate f"{dict}" behavior from MangaDialougeDatasetCreator
    user_content_str = str(user_content_dict)
    
    return {
        "messages": [
            {
                "role": "user",
                "content": user_content_str
            },
            {
                "role": "assistant",
                "content": en
            }
        ]
    }

from datasets import concatenate_datasets

concatenated_dataset = concatenate_datasets([jesc_dataset, flores_ds, ntrex_ds, opus100_eval_ds])

import json
from tqdm.auto import tqdm

BATCH_SIZE = 50000  # Larger batches for 2GB dataset

with open(SAVE_FILE, "w", buffering=16*1024*1024) as f:  # 16MB buffer
    batch = []
    for item in tqdm(concatenated_dataset):
        ja = item['src_text']
        en = item['tgt_text']
        batch.append(json.dumps(format_opus_to_manga_style(ja, en)))
        
        if len(batch) >= BATCH_SIZE:
            f.write('\n'.join(batch) + '\n')
            batch = []
    
    if batch:
        f.write('\n'.join(batch) + '\n')