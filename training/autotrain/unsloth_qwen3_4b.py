#!/usr/bin/env python3
"""
Zen Coder Training: Qwen3 4B with Unsloth (2x faster)
Run: python unsloth_qwen3_4b.py
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# Configuration
# ============================================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET = "hanzoai/zen-agentic-dataset"  # Or local path
OUTPUT_DIR = "./output/zen-coder-4b-qwen3"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True  # QLoRA - fits in 8GB VRAM

# Training params
EPOCHS = 1  # Test run
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4  # Unsloth recommends higher LR

# ============================================
# Load Model with Unsloth
# ============================================
print(f"Loading {MODEL_NAME} with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (bfloat16 for newer GPUs)
    load_in_4bit=LOAD_IN_4BIT,
)

# ============================================
# Add LoRA Adapters
# ============================================
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=42,
)

# ============================================
# Load Dataset
# ============================================
print(f"Loading dataset: {DATASET}")
dataset = load_dataset(DATASET, split="train")

# Format for chat
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
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
    packing=True,  # Unsloth packing for efficiency
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.03,
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

print("Starting training...")
trainer.train()

# ============================================
# Save Model
# ============================================
print(f"Saving to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: Push to HuggingFace
# model.push_to_hub("zenlm/zen-coder-4b-qwen3")

print("Done!")
