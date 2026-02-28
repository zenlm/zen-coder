#!/usr/bin/env python3
"""
Zen Coder ULTRA: Kimi K2 Thinking (1T MoE) Fine-tuning
REQUIRES: 8xH200+ cluster (400GB+ VRAM) or multi-node

Model: moonshotai/Kimi-K2-Instruct
- 1 Trillion parameters (MoE with 32 experts)
- 32B activated parameters per token
- 128K context window
- MIT License (!)

This is the LARGEST model in Zen Coder lineup.
Run on HF with 8xH200 cluster: ~$35/hr × 12-24hrs = ~$420-840
"""
import os
import json
import torch

# ============================================
# Hardware Check
# ============================================
print("=" * 60)
print("Kimi K2 Thinking (1T) - Zen Coder ULTRA Training")
print("=" * 60)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    print(f"GPUs: {num_gpus}")
    print(f"Total VRAM: {total_vram / 1e9:.1f} GB")

    if total_vram < 400e9:
        print("\n⚠️  WARNING: Kimi K2 1T requires ~400GB+ VRAM for QLoRA training")
        print("Options:")
        print("  1. Use multi-node cluster (2+ 8xH200 nodes)")
        print("  2. Use DeepSpeed ZeRO-3 with NVMe offload")
        print("  3. Use smaller model (GLM-4.5 358B)")
        print("\nFor single 8xH200 node, try GLM-4.5 instead.")
        # Don't exit - may be running with DeepSpeed ZeRO-3
else:
    print("No CUDA available. Kimi K2 requires multi-GPU cluster.")
    exit(1)

# ============================================
# Configuration
# ============================================
MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DATASET = "hanzoai/zen-agentic-dataset"
OUTPUT_DIR = "./output/zen-coder-ultra-kimi-k2"
MAX_SEQ_LENGTH = 8192  # 128K supported, use 8K for training

# Training params - very conservative for 1T
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 32  # Effective batch = 32
LR = 1e-6  # Extremely low LR for largest model

# ============================================
# DeepSpeed ZeRO-3 + NVMe Offload Config
# ============================================
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False,
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "gradient_accumulation_steps": GRAD_ACCUM,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
}

# ============================================
# HuggingFace Autotrain (Multi-Node)
# ============================================
AUTOTRAIN_CONFIG = """
# Kimi K2 requires multi-node training
# For HuggingFace multi-node with 2x 8xH200:
#
# autotrain llm \\
#   --model moonshotai/Kimi-K2-Instruct \\
#   --data-path hanzoai/zen-agentic-dataset \\
#   --train-split train \\
#   --valid-split valid \\
#   --chat-template chatml \\
#   --text-column messages \\
#   --lr 1e-6 \\
#   --epochs 1 \\
#   --batch-size 1 \\
#   --gradient-accumulation 32 \\
#   --block-size 8192 \\
#   --peft \\
#   --quantization int4 \\
#   --lora-r 8 \\
#   --lora-alpha 16 \\
#   --target-modules q_proj,k_proj,v_proj,o_proj \\
#   --mixed-precision bf16 \\
#   --gradient-checkpointing \\
#   --project-name zen-coder-ultra-kimi-k2

# Note: Kimi K2 is the largest model (1T params)
# Expected training time: 12-24 hours on 2x 8xH200 (~$840-1680)
# Consider using GLM-4.5 (358B) for faster iteration
"""

print(AUTOTRAIN_CONFIG)
print("\n" + "=" * 60)
print("For local multi-node training with DeepSpeed:")
print("deepspeed --num_gpus=8 --num_nodes=2 unsloth_kimi_k2_1t.py")
print("=" * 60)

# ============================================
# Kimi K2 Architecture Notes
# ============================================
ARCHITECTURE_NOTES = """
Kimi K2 Architecture (from paper):
- Total Parameters: 1,043B (1 Trillion)
- Activated Parameters: 32B per token
- MoE Layers: 60
- Dense Layers: 1
- # Experts: 384 total, 8 active per token
- Shared Experts: 1
- Hidden Dim: 7168
- Attention Heads: 64
- KV Heads: 64

Key differences from GLM-4.5:
- Kimi K2 is larger (1T vs 355B)
- GLM-4.5 has more attention heads per dim (96 vs 64)
- GLM-4.5 uses QK-Norm for stability
- GLM-4.5 has deeper layers (89 MoE layers vs 60)

For benchmarking:
- SWE-bench Verified: Kimi K2 65.4%, GLM-4.5 64.2%
- TAU-bench: Kimi K2 ~62%, GLM-4.5 ~70%
- AIME-24: Both ~90%+
"""

print(ARCHITECTURE_NOTES)

# ============================================
# Training Code (Multi-node required)
# ============================================
if __name__ == "__main__" and num_gpus >= 8:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print(f"\nLoading {MODEL_NAME}...")
    print("This will take significant time due to model size...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    # Kimi K2 may require trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config - extremely conservative for 1T
    lora_config = LoraConfig(
        r=8,  # Very low rank for 1T model
        lora_alpha=16,
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
            save_steps=100,
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

    print("\nStarting Kimi K2 1T training...")
    print("Expected time: 12-24 hours on multi-node cluster")
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
