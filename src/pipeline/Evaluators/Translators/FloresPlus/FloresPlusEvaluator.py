import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore
from torchmetrics.text import CHRFScore
from torchmetrics.text.bert import BERTScore
from datasets import load_dataset
from tqdm.auto import tqdm
from tabulate import tabulate
from transformers import pipeline
import time
import gc
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results
from TranslationEvaluator import TranslationEvaluator
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

class FloresPlusEvaluator(TranslationEvaluator):
    def __init__(self, hf_token=None):
        self.hf_token = hf_token

    def get_dataset(self):
        ds_jpn = load_dataset("openlanguagedata/flores_plus", "jpn_Jpan", split = 'devtest')
        ds_eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split = 'devtest')
        flores_ds = FloresDataset(ds_jpn, ds_eng)
        return flores_ds
        
    def get_dataloader(self, batch_size=1):
        flores_ds = self.get_dataset()
        flores_dataloader = DataLoader(flores_ds, batch_size=batch_size, shuffle=False)
        return flores_dataloader

    def get_evaluator_name(self):
        return "FloresPlus"


