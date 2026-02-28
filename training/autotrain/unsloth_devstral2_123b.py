#!/usr/bin/env python3
"""
Zen Coder 123B: Devstral 2 (123B) Fine-tuning
REQUIRES: 8xH200 (128GB+ VRAM with QLoRA)

Model: mistralai/Devstral-2-123B-Instruct-2512
- 123B parameters (dense architecture)
- 256K context window
- QLoRA fits in 128GB VRAM (single 8xH200 node)

Run on HF with 8xH200: ~$35/hr × 4-8hrs = ~$140-280
"""
import os
import json
import torch

# ============================================
# Hardware Check
# ============================================
print("=" * 60)
print("Devstral 2 (123B) - Zen Coder 123B Training")
print("=" * 60)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    print(f"GPUs: {num_gpus}")
    print(f"Total VRAM: {total_vram / 1e9:.1f} GB")

    if total_vram < 128e9:
        print("\n⚠️  WARNING: Devstral 2 123B requires ~128GB VRAM for QLoRA training")
        print("Options:")
        print("  1. Use HuggingFace 8xH200 ($35/hr)")
        print("  2. Use DeepSpeed ZeRO-3 with aggressive offloading")
        print("  3. Use smaller model (Devstral Small 24B)")
        exit(1)
else:
    print("No CUDA available. Devstral 2 requires multi-GPU setup.")
    exit(1)

# ============================================
# Configuration
# ============================================
MODEL_NAME = "mistralai/Devstral-2-123B-Instruct-2512"
DATASET = "hanzoai/zen-agentic-dataset"
OUTPUT_DIR = "./output/zen-coder-123b-devstral2"
MAX_SEQ_LENGTH = 8192  # 256K supported, use 8K for training

# Training params - conservative for 123B
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 16  # Effective batch = 16
LR = 5e-5  # Very low LR for large model

# ============================================
# DeepSpeed Config for 123B
# ============================================
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
    },
    "gradient_accumulation_steps": GRAD_ACCUM,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
}

# ============================================
# HuggingFace Autotrain Config (Recommended)
# ============================================
AUTOTRAIN_CONFIG = """
# Run on HuggingFace with 8xH200:
# autotrain llm \\
#   --model mistralai/Devstral-2-123B-Instruct-2512 \\
#   --data-path hanzoai/zen-agentic-dataset \\
#   --train-split train \\
#   --valid-split valid \\
#   --chat-template mistral \\
#   --text-column messages \\
#   --lr 5e-5 \\
#   --epochs 1 \\
#   --batch-size 1 \\
#   --gradient-accumulation 16 \\
#   --block-size 8192 \\
#   --peft \\
#   --quantization int4 \\
#   --lora-r 16 \\
#   --lora-alpha 32 \\
#   --target-modules all-linear \\
#   --mixed-precision bf16 \\
#   --gradient-checkpointing \\
#   --project-name zen-coder-123b-devstral2
"""

print(AUTOTRAIN_CONFIG)
print("\n" + "=" * 60)
print("For local multi-GPU training with DeepSpeed:")
print("deepspeed --num_gpus=8 unsloth_devstral2_123b.py")
print("=" * 60)

# ============================================
# Training Code
# ============================================
if __name__ == "__main__" and num_gpus >= 4:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print(f"\nLoading {MODEL_NAME}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config - conservative for 123B
    lora_config = LoraConfig(
        r=16,  # Low rank for large model
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    print(f"\nLoading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split="train")

    def format_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_func, batched=True)

    # Training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="tensorboard",
            gradient_checkpointing=True,
            deepspeed=DEEPSPEED_CONFIG,
        ),
    )

    print("\nStarting Devstral 2 123B training...")
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
