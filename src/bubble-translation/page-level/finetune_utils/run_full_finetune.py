import json
import os
import sys
import unsloth
import backoff
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from base_train_config import TrainingConfig
from trainer import sft_train
from finetune_util import load_dataset_from_jsonl
from dotenv import load_dotenv

import weave

# Load environment variables from .env file
load_dotenv()

def load_model_for_full_finetune(model_id, load_in_4bit=False, max_seq_length=2048):
    """Load model and tokenizer for full finetuning"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
    )
    
    # Configure tokenizer padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with appropriate dtype based on hardware
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        torch_dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Set max length for the model
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max_seq_length
    
    return model, tokenizer

def train(training_cfg):
    """Prepare model, call training function, and push to hub"""
    model, tokenizer = load_model_for_full_finetune(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length
    )

    print("Setting up model for full finetuning")
    
    # Prepare the dataset using memory-efficient loading
    training_files = training_cfg.training_file
    if isinstance(training_files, str):
        training_files = [training_files]
    
    datasets = []
    for tf in training_files:
        ds = load_dataset_from_jsonl(tf, training_cfg.loss)
        print(f"Loaded {len(ds)} samples from {tf}")
        datasets.append(ds)
    
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total samples after merging: {len(dataset)}")
    
    # Shuffle with buffer to limit memory usage
    dataset = dataset.shuffle(seed=training_cfg.seed)
    
    if training_cfg.test_file:
        test_dataset = load_dataset_from_jsonl(training_cfg.test_file, training_cfg.loss)
        split = dataset.train_test_split(test_size=0.1, seed=training_cfg.seed)
        dataset = split["train"]
        # this is so our train set is the same when we have a different test set!
        dont_use_me_dataset = split["test"]
    else:
        # Split 10% of train data for testing when no test set provided
        # Use seed from training config to make the split deterministic
        split = dataset.train_test_split(test_size=0.1, seed=training_cfg.seed)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    print(f"Pushing full model and tokenizer to {finetuned_model_id}...")
    model.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    tokenizer.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    print("Successfully pushed full model and tokenizer!")


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    
    # Override is_peft setting for full finetuning
    config["is_peft"] = False
    
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1]) 