"""
sft.py

This module performs supervised fine-tuning (SFT) by loading the filtered dataset,
applying a chat template, configuring and applying LoRA for efficiency, and training the model.
Safety policies are loaded from a YAML configuration file.
This version uses a single GPU.
"""

import os
import yaml
import torch

from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from unsloth.chat_templates import standardize_sharegpt, get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

def load_policies(policies_config_path):
    with open(policies_config_path, "r") as f:
        policies = yaml.safe_load(f)
    return policies

def run_sft_stage(args):
    """
    Entry point for SFT training on a single GPU using the filtered SFT dataset.
    """
    # Load policies from YAML file.
    print("Loading policies...")
    policies = load_policies(args['policies_config'] if isinstance(args, dict) else args.policies_config)
    
    # Load the filtered SFT data and print some debug information
    print("Loading filtered SFT data from disk...")
    from datasets import load_from_disk
    dataset = load_from_disk("filtered_sft_data")
    print("Dataset keys:", dataset.column_names)
    
    # DEBUG: Print information about the dataset
    print(f"Dataset size: {len(dataset)}")
    print("First few examples:")
    for i in range(min(3, len(dataset))):
        print(f"Example {i}:")
        if "prompt" in dataset.column_names:
            print(f"  Prompt: {dataset[i]['prompt'][:100]}...")
        if "answer" in dataset.column_names:
            print(f"  Answer: {dataset[i]['answer'][:100]}...")
    
    # The filtered dataset is expected to have keys: "prompt", "category", "cot", "answer", "judge_score".
    
    # Load model and tokenizer first
    print("Loading model and tokenizer...")
    max_seq_length = args['max_seq_length'] if isinstance(args, dict) else args.max_seq_length
    dtype = None  # Let unsloth automatically pick the best dtype.
    load_in_4bit = args['load_in_4bit'] if isinstance(args, dict) else args.load_in_4bit

    # Load the base model and tokenizer.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args['sft_model'] if isinstance(args, dict) else args.sft_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Apply the chat template (using llama-3.1 style).
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    # Format data using the proper chat message structure
    def formatting_filtered_func(examples):
        messages = []
        for i in range(len(examples["prompt"])):
            # Get the final, complete prompt (not the incremental versions)
            prompt = examples['prompt'][i]
            
            # Get only the actual answer without the "Final Answer:" marker
            answer = examples['answer'][i].replace("Final Answer:", "").strip()
            
            # Skip samples with empty answers
            if not answer:
                continue
            
            messages.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ])
        return {"messages": messages}
    
    print("Formatting dataset with message structure...")
    dataset = dataset.map(formatting_filtered_func, batched=True)
    
    # Print some debug information
    print("Dataset keys after formatting:", dataset.column_names)
    if "messages" not in dataset.column_names:
        raise ValueError("'messages' field not found in dataset after formatting. Available fields: " + str(dataset.column_names))
    
    # Verify the first example has correct format
    print("=== Example Formatted Messages ===")
    print(dataset[0]["messages"])
    
    # Apply chat template to convert the messages into a single text field
    def apply_chat_template(example):
        example_messages = example["messages"]
        formatted_chat = tokenizer.apply_chat_template(
            example_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_chat}
    
    print("Applying chat template to convert messages to text format...")
    dataset = dataset.map(apply_chat_template)
    
    # Create a clean dataset with only the fields needed for training
    clean_dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    
    # Initialize LoRA for efficient fine-tuning.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args['seed'] if isinstance(args, dict) else args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Unsloth needs the tokenizer to be attached to the model
    model.tokenizer = tokenizer
    
    # Setup Trainer.
    training_args = TrainingArguments(
        per_device_train_batch_size=args['sft_batch_size'] if isinstance(args, dict) else args.sft_batch_size,
        gradient_accumulation_steps=args['sft_grad_accum_steps'] if isinstance(args, dict) else args.sft_grad_accum_steps,
        warmup_steps=5,
        max_steps=args['sft_max_steps'] if isinstance(args, dict) else args.sft_max_steps,
        learning_rate=args['sft_learning_rate'] if isinstance(args, dict) else args.sft_learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args['seed'] if isinstance(args, dict) else args.seed,
        output_dir="outputs/sft",
        report_to="none"
    )

    # Use SFTTrainer from TRL for instruction tuning
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=clean_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model.
    sft_model_path = "sft_model"
    model.save_pretrained(sft_model_path)
    tokenizer.save_pretrained(sft_model_path)
    print("Supervised fine-tuning completed. Model saved at:", sft_model_path)