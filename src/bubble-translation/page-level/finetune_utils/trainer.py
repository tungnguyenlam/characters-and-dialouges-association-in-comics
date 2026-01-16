import os
import torch
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


# Store original before patching
_original_trainer_compute_loss = Trainer.compute_loss

def _fixed_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Fixed compute_loss that avoids inplace modification of Unsloth's custom backward tensors.
    This prevents: RuntimeError: Output 0 of UnslothFusedLossBackward is a view and is being modified inplace.
    
    The fix: Clone the loss tensor before the inplace *= operation in transformers 4.57+
    """
    # Temporarily restore original to call it
    Trainer.compute_loss = _original_trainer_compute_loss
    try:
        # Call original compute_loss
        result = _original_trainer_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        # Re-apply our patch
        Trainer.compute_loss = _fixed_compute_loss
    
    # The issue is that the original does loss *= scalar (inplace)
    # But we've already called it, so the error would have occurred
    # We need a different approach - let's just avoid the *= entirely
    
    return result

# Actually, the above won't work. Let me do it properly by reimplementing without the inplace op
def _fixed_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Fixed compute_loss that avoids inplace modification of Unsloth's custom backward tensors.
    Reimplements transformers.Trainer.compute_loss without inplace operations.
    """
    # Handle label smoothing
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    
    # Forward pass
    outputs = model(**inputs)
    
    # Save past state if needed
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    # Extract loss
    if labels is not None:
        if isinstance(outputs, dict):
            loss_key = "loss" if "loss" in outputs else None
            loss = outputs[loss_key] if loss_key else None
        else:
            loss = outputs[0] if len(outputs) > 0 else None
            
        if loss is None:
            raise ValueError("Model did not return loss")
            
        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}."
            )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    # Clone loss BEFORE any operations - this is the fix for Unsloth
    loss = loss.clone() if torch.is_tensor(loss) else loss
    
    # Scale loss (non-inplace operation using * not *=)
    if self.args.n_gpu > 1:
        loss = loss * self.args.n_gpu  
    else:
        loss = loss * self.accelerator.num_processes
    
    return (loss, outputs) if return_outputs else loss

Trainer.compute_loss = _fixed_compute_loss


# Use regular SFTTrainer since we patched the base Trainer class
PatchedSFTTrainer = SFTTrainer


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    import wandb
    wandb.init(
        project="clarifying-em",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=["wandb"],
            num_train_epochs=training_cfg.epochs,
            push_to_hub=True,
            hub_model_id=training_cfg.finetuned_model_id,
            hub_strategy="every_save",  # Options: "end", "every_save", "checkpoint"
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            **kwargs,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=getattr(training_cfg, 'early_stopping_patience', 5))],
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            PatchedSFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = PatchedSFTTrainer(**trainer_kwargs)
    return trainer
