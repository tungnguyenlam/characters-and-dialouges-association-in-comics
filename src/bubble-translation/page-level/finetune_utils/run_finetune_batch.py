import json
import os
import sys

import backoff
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel

from base_train_config import TrainingConfig
from trainer import sft_train
from finetune_util import load_dataset_from_jsonl, load_model_and_tokenizer
from dotenv import load_dotenv

import weave

# Load environment variables from .env file
load_dotenv()

def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        layers_to_transform=training_cfg.layers_to_transform,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    dataset = load_dataset_from_jsonl(training_cfg.training_file, training_cfg.loss)
    print(f"Loaded {len(dataset)} samples from {training_cfg.training_file}")
    
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
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")

    if hasattr(training_cfg, 'output_dir') and training_cfg.output_dir:
        import shutil
        if os.path.exists(training_cfg.output_dir):
            print(f"Deleting model from disk: {training_cfg.output_dir}")
            shutil.rmtree(training_cfg.output_dir)
            print("Model deleted from disk")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    # Validate configuration
    if training_cfg.merge_before_push and training_cfg.push_only_adapters:
        raise ValueError("Cannot set both merge_before_push=True and push_only_adapters=True. "
                        "After merging, the model no longer has LoRA adapters to push separately.")
    
    # First merge if requested
    if training_cfg.merge_before_push:
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
        print("Successfully merged weights!")
    
    # Then push based on push_only_adapters setting
    if training_cfg.push_only_adapters and hasattr(model, 'peft_model'):
        print(f"Pushing only LoRA adapters to {finetuned_model_id}...")
        # Only push the LoRA adapters
        model.peft_model.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print("Successfully pushed LoRA adapters!")
    else:
        print(f"Pushing {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer to {finetuned_model_id}...")
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        print(f"Successfully pushed {'merged ' if training_cfg.merge_before_push else 'full '}model and tokenizer!")

    try:
        del model, tokenizer
    except Exception as e:
        print(f"Error deleting model and tokenizer: {e}")

def main(config_dir: str, delete_model: bool = False):
    import gc
    import glob
    import shutil
    import torch

    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    
    for config_file in config_files:

        print(f"Processing: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        training_config = TrainingConfig(**config)
        train(training_config)

        if delete_model:
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(hf_cache_dir):
                print(f"Clearing HuggingFace hub cache directory: {hf_cache_dir}")
                for item in os.listdir(hf_cache_dir):
                    item_path = os.path.join(hf_cache_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        print(f"Removed: {item_path}")
                    except Exception as e:
                        print(f"Error removing {item_path}: {e}")
                print("HuggingFace hub cache cleared")
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    delete_flag = len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'yes']
    main(sys.argv[1], delete_flag)