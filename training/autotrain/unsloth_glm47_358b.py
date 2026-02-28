#!/usr/bin/env python3
"""
Zen Coder MAX: GLM-4.7 (358B MoE) Fine-tuning
REQUIRES: 8xH200 or 2xA100-80GB (~180GB VRAM with QLoRA)

Model: zai-org/GLM-4.7
- 358B total parameters (MoE architecture)
- 200K context window, 128K output capacity
- BF16 weights: ~716 GB
- QLoRA (Q4): ~180 GB VRAM required

Run on HF with 8xH200: ~$35/hr × 6-12hrs = ~$210-420

GLM-4.7 Benchmarks (Dec 2025):
- LiveCodeBench V6: 84.9% (SOTA)
- SWE-bench Verified: 73.8%
- AIME 2025: 95.7%
- Terminal-Bench: 37.5%

Note: For 128GB VRAM constraint, use Devstral 123B instead.
"""
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset

# Check GPU configuration
print("=" * 60)
print("GLM-4.7 (358B) - Zen Coder MAX Training")
print("=" * 60)

# Hardware check
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    print(f"GPUs: {num_gpus}")
    print(f"Total VRAM: {total_vram / 1e9:.1f} GB")

    if total_vram < 180e9:
        print("\n⚠️  WARNING: GLM-4.7 requires ~180GB VRAM for QLoRA training")
        print("Options:")
        print("  1. Use HuggingFace 8xH200 ($35/hr)")
        print("  2. Use 2xA100-80GB (160GB)")
        print("  3. Use DeepSpeed ZeRO-3 with aggressive CPU offload")
        print("  4. Use Devstral 123B instead (fits 128GB)")
        # Don't exit - may work with DeepSpeed offloading
else:
    print("No CUDA available. GLM-4.7 requires multi-GPU setup.")
    exit(1)

# ============================================
# Configuration
# ============================================
MODEL_NAME = "zai-org/GLM-4.7"
DATASET = "hanzoai/zen-agentic-dataset"
OUTPUT_DIR = "./output/zen-coder-max-glm47"
MAX_SEQ_LENGTH = 8192  # 200K supported, use 8K for training

# ============================================
# DeepSpeed Config for Multi-GPU
# ============================================
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": 16,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 1,
}

# ============================================
# For HuggingFace Autotrain (Recommended)
# ============================================
AUTOTRAIN_CONFIG = """
# Run on HuggingFace with 8xH200:
# autotrain llm \\
#   --model zai-org/GLM-4.7 \\
#   --data-path hanzoai/zen-agentic-dataset \\
#   --train-split train \\
#   --valid-split valid \\
#   --chat-template glm4 \\
#   --text-column messages \\
#   --lr 5e-6 \\
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
#   --project-name zen-coder-max-glm47
"""

print(AUTOTRAIN_CONFIG)
print("\n" + "=" * 60)
print("For local multi-GPU training, use DeepSpeed:")
print("deepspeed --num_gpus=8 unsloth_glm47_358b.py")
print("=" * 60)

# ============================================
# Training Code (requires 8xH100+)
# ============================================
if __name__ == "__main__" and num_gpus >= 8:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load with Unsloth (if supported) or transformers
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Very low rank for 358B
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except Exception as e:
        print(f"Unsloth not supported for GLM-4.7, using transformers: {e}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    # Dataset
    dataset = load_dataset(DATASET, split="train")

    def format_func(examples):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in examples["messages"]]}

    dataset = dataset.map(format_func, batched=True)

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            deepspeed=DEEPSPEED_CONFIG,
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
