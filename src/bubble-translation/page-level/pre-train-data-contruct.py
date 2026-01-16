import os
import sys
from torch.utils.data import Dataset

cwd = os.getcwd()
if not cwd.endswith("group-project-b3"):
    raise ValueError("Please run this script in the root directory of the project")

SAVE_DIR = os.path.join(cwd, "data", "translation", "pre-train")
SAVE_FILE = os.path.join(SAVE_DIR, "pre-train-data.json")
os.makedirs(SAVE_DIR, exist_ok=True)

# Create data loader for JESC 2019

jesc_path = os.path.join(cwd, "data", "jesc", "dataset.csv")

import pandas as pd

jesc_df= pd.read_csv(jesc_path)
jap=(jesc_df["ja"].tolist())
en= (jesc_df["en"].tolist())

class TextDataset(Dataset):
    def __init__(self, texts, trans):
        self.texts=texts
        self.trans= trans

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "src": self.texts[idx],
            "tgt": self.trans[idx]
        }

jesc_dataset= TextDataset(jap,en)

class FloresDataset(Dataset):
    def __init__(self, ds_src, ds_tgt):
        """
        ds_src: Japanese dataset
        ds_tgt: English dataset (or target language)
        They should be aligned by index
        """
        self.ds_src = ds_src
        self.ds_tgt = ds_tgt
        assert len(ds_src) == len(ds_tgt), "Source and target datasets must have same length"
    
    def __len__(self):
        return len(self.ds_src)
    
    def __getitem__(self, index):
        return {
            "src_text": self.ds_src[index]['text'],
            "tgt_text": self.ds_tgt[index]['text']
        }

ds_jpn = load_dataset("openlanguagedata/flores_plus", "jpn_Jpan", split = 'dev')
ds_eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split = 'dev')
flores_ds = FloresDataset(ds_jpn, ds_eng)

class NTRexDataset(Dataset):
    def __init__(self, ds_src, ds_tgt):
        """
        ds_src: Source language dataset (e.g., Japanese)
        ds_tgt: Target language dataset (e.g., English)
        They should be aligned by index.
        """
        self.ds_src = ds_src
        self.ds_tgt = ds_tgt
        
        # Verify that both datasets have the same number of examples
        assert len(ds_src) == len(ds_tgt), "Source and target datasets must have same length"
    
    def __len__(self):
        return len(self.ds_src)
    
    def __getitem__(self, index):
        # Based on your finding: 
        # ds_src (config="ja") has a column 'text'
        # ds_tgt (config="en") has a column 'text'
        return {
            "src_text": self.ds_src[index]['text'],
            "tgt_text": self.ds_tgt[index]['text']
        }

ds_jpn = load_dataset("xianf/NTREX","ja", split='train', token=self.hf_token)
ds_eng = load_dataset("xianf/NTREX","en", split='train', token=self.hf_token)
ntrex_ds = NTRexDataset(ds_jpn, ds_eng)

class Opus100EvaluationDataset(Dataset):
    def __init__(self, ds, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        return {
            "src_text": src_text,
            "tgt_text": tgt_text
        }

ds_raw = load_dataset('opus100', 'en-ja', split='train', token=self.hf_token)

opus100_eval_ds = Opus100EvaluationDataset(ds_raw, 'ja', 'en')

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

from dataset import concatenate_datasets

concatenated_dataset = concatenate_datasets([jesc_dataset, flores_ds, ntrex_ds, opus100_eval_ds])

import json
from tqdm.auto import tqdm

for ja, en in tqdm(concatenated_dataset):
    json.dump(format_opus_to_manga_style(ja, en), open(SAVE_FILE, "a"))
    
