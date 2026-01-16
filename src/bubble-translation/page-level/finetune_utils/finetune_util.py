import json
import os

import torch
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from datasets import load_dataset

load_dotenv()


def load_model_and_tokenizer(model_id, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except Exception as e:
        print(f"Error checking if model is PEFT: {e}, continuing...")
        pass
    return is_peft


def load_jsonl(file_id):
    """Legacy function - prefer load_dataset_from_jsonl for memory efficiency"""
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def load_dataset_from_jsonl(file_path, loss_type="sft"):
    """
    Memory-efficient dataset loading using HuggingFace's load_dataset.
    Uses Arrow format with memory mapping instead of loading into Python lists.
    """
    ds = load_dataset("json", data_files=file_path, split="train")
    
    if loss_type == "sft":
        ds = ds.select_columns(["messages"])
    
    return ds
