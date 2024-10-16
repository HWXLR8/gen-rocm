#!/usr/bin/env python3

import torch
import transformers
from datasets import load_dataset, Features, Value
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

# load dataset
features = Features({ "text": Value("string") })
dataset = load_dataset("text", data_files={"train": "shakespeare.txt"}, sample_by="paragraph")['train']

# load base model
base_model_id = "meta-llama/Meta-Llama-3-8B"
login() # for llama 3 you need permission (it's given automatically when you request, but you need to login)
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

# tokenization
tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
tokenizer.pad_token = tokenizer.eos_token

max_length = 256

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        prompt['text'],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

# set up LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# train
project = "shakespeare-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5, # we want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps", # Save the model checkpoint every logging step
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
