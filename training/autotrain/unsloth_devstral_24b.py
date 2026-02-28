#!/usr/bin/env python3
"""
Zen Coder Training: Devstral Small 2 (24B) with Unsloth
Best open-source coding model (Apache 2.0)
Requires: ~24GB VRAM with QLoRA (RTX 4090, A100)
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# Configuration
# ============================================
MODEL_NAME = "mistralai/Devstral-Small-2-24B-Instruct-2512"
DATASET = "hanzoai/zen-agentic-dataset"
OUTPUT_DIR = "./output/zen-coder-devstral-24b"
MAX_SEQ_LENGTH = 8192  # Devstral supports 256K, use 8K for training
LOAD_IN_4BIT = True  # QLoRA required for 24B

# Training params - conservative for 24B
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8  # Effective batch = 8
LR = 1e-4  # Lower LR for larger model

# ============================================
# Load Model
# ============================================
print(f"Loading {MODEL_NAME} with Unsloth (QLoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# ============================================
# LoRA Configuration
# ============================================
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Lower rank for 24B model
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================
# Dataset
# ============================================
print(f"Loading dataset: {DATASET}")
dataset = load_dataset(DATASET, split="train")

def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        # Mistral chat format
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================
# Training
# ============================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=4,
    packing=True,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="tensorboard",
    ),
)

print("Starting Devstral Small 2 training...")
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
