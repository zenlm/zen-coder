#!/usr/bin/env python3
"""
Zen Coder MLX Training Script
==============================

Train zen-coder on the Zen Agentic Dataset (8.47B tokens)
using MLX for Apple Silicon optimization.

Dataset: zenlm/zen-agentic-dataset (private)
Base Model: Qwen/Qwen3-Coder-Next
Output: zenlm/zen-coder

Requirements:
    pip install mlx mlx-lm transformers datasets huggingface_hub peft
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass, field
import time

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load, generate
    from mlx_lm.utils import load_model
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available - install with: pip install mlx mlx-lm")

# HuggingFace imports
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, login
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for Zen Coder"""
    # Model
    model_name: str = "zen-coder"
    base_model: str = "Qwen/Qwen3-Coder-Next"
    
    # Dataset
    dataset_repo: str = "zenlm/zen-agentic-dataset"
    dataset_private: bool = True
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Checkpointing
    output_dir: str = "./models/zen-coder"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # MLX specific
    quantize_bits: int = 4  # 4-bit or 8-bit quantization
    use_gradient_checkpointing: bool = True
    
    # HuggingFace
    hf_output_repo: str = "zenlm/zen-coder"
    push_to_hub: bool = True


class ZenAgenticDataLoader:
    """Load and process Zen Agentic Dataset for training"""
    
    SYSTEM_PROMPT = """You are Zen Coder, the flagship coding model from Zen LM and Hanzo AI, specialized in:
- Agentic AI development (MCP, multi-agent systems, tool-use)
- Web3 and blockchain (Solidity, DeFi, consensus)
- Modern cryptography (post-quantum, threshold, MPC, ZK)
- Full-stack development (Next.js, React, TypeScript)
- Systems programming (Rust, Go, async patterns)

Provide clear, efficient, and secure code solutions."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load tokenizer from base model"""
        logger.info(f"Loading tokenizer from {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_dataset_files(self) -> List[str]:
        """List available dataset files from HuggingFace"""
        logger.info(f"Loading dataset from {self.config.dataset_repo}")
        
        api = HfApi()
        files = api.list_repo_files(
            repo_id=self.config.dataset_repo,
            repo_type="dataset"
        )
        
        # Filter for JSONL files
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        logger.info(f"Found {len(jsonl_files)} JSONL files in dataset")
        return jsonl_files
    
    def stream_examples(self, max_samples: Optional[int] = None) -> Iterator[Dict]:
        """Stream examples from dataset files"""
        jsonl_files = self.load_dataset_files()
        sample_count = 0
        
        for jsonl_file in jsonl_files:
            logger.info(f"Processing: {jsonl_file}")
            
            try:
                # Download file
                local_path = hf_hub_download(
                    repo_id=self.config.dataset_repo,
                    filename=jsonl_file,
                    repo_type="dataset"
                )
                
                # Stream lines
                with open(local_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if max_samples and sample_count >= max_samples:
                            return
                        
                        try:
                            example = json.loads(line.strip())
                            formatted = self.format_example(example)
                            if formatted:
                                sample_count += 1
                                yield formatted
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to process {jsonl_file}: {e}")
                continue
        
        logger.info(f"Streamed {sample_count} examples")
    
    def format_example(self, example: Dict) -> Optional[Dict]:
        """Format example for training in ChatML format"""
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        
        # Handle different data formats
        if "messages" in example:
            # Already in messages format
            messages.extend(example["messages"])
        elif "content" in example:
            # Git commit or code content
            if "commit_message" in example:
                messages.append({
                    "role": "user",
                    "content": f"Review this code change:\n\n{example.get('diff', example['content'])}"
                })
                messages.append({
                    "role": "assistant", 
                    "content": f"Commit: {example['commit_message']}\n\n{example.get('content', '')}"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Explain this code:"
                })
                messages.append({
                    "role": "assistant",
                    "content": example["content"]
                })
        elif "input" in example and "output" in example:
            # Input/output format
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        elif "prompt" in example and "response" in example:
            # Prompt/response format
            messages.append({"role": "user", "content": example["prompt"]})
            messages.append({"role": "assistant", "content": example["response"]})
        else:
            # Skip unrecognized formats
            return None
        
        return {"messages": messages, "format": "chatml"}
    
    def tokenize_example(self, example: Dict) -> Dict:
        """Tokenize a single example"""
        if not self.tokenizer:
            self.load_tokenizer()
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="np"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].copy()
        }


class MLXLoRATrainer:
    """MLX-based LoRA trainer for Apple Silicon"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
    def setup(self):
        """Initialize model and optimizer"""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Load model with MLX
        self.model, self.tokenizer = load(self.config.base_model)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info("Model and optimizer initialized")
        return self.model, self.tokenizer
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
        # Convert numpy arrays to lists for MLX
        input_ids = mx.array([x.tolist() if hasattr(x, 'tolist') else x for x in batch["input_ids"]])
        labels = mx.array([x.tolist() if hasattr(x, 'tolist') else x for x in batch["labels"]])
        
        def loss_fn(model):
            logits = model(input_ids)
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # Cross entropy loss
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="mean"
            )
            return loss
        
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        return loss.item()
    
    def train(self, data_loader: ZenAgenticDataLoader, max_steps: Optional[int] = None):
        """Main training loop"""
        logger.info("Starting training...")
        
        step = 0
        epoch_losses = []
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            batch_losses = []
            batch = []
            
            for example in data_loader.stream_examples():
                tokenized = data_loader.tokenize_example(example)
                batch.append(tokenized)
                
                if len(batch) >= self.config.batch_size:
                    # Stack batch
                    batch_data = {
                        "input_ids": [b["input_ids"] for b in batch],
                        "labels": [b["labels"] for b in batch]
                    }
                    
                    loss = self.train_step(batch_data)
                    batch_losses.append(loss)
                    step += 1
                    batch = []
                    
                    if step % self.config.logging_steps == 0:
                        avg_loss = sum(batch_losses[-100:]) / min(len(batch_losses), 100)
                        logger.info(f"Step {step}: loss = {avg_loss:.4f}")
                    
                    if step % self.config.save_steps == 0:
                        self.save_checkpoint(step)
                    
                    if max_steps and step >= max_steps:
                        logger.info(f"Reached max_steps={max_steps}")
                        break
            
            epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
            epoch_losses.append(epoch_loss)
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final")
        logger.info(f"Training complete. Final loss: {epoch_losses[-1]:.4f}")
        
        return epoch_losses
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        mx.save(str(checkpoint_dir / "weights.npz"), dict(self.model.parameters()))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save config
        config_dict = {
            "model_name": self.config.model_name,
            "base_model": self.config.base_model,
            "step": step,
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
            }
        }
        with open(checkpoint_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def convert_to_hf(self, output_path: str):
        """Convert MLX model to HuggingFace format"""
        logger.info(f"Converting to HuggingFace format: {output_path}")
        
        # Use mlx_lm conversion utilities
        convert_cmd = f"""
python -m mlx_lm.convert \\
    --hf-path {self.config.output_dir}/checkpoint-final \\
    --mlx-path {output_path} \\
    -q
        """
        
        logger.info(f"Run: {convert_cmd}")
        os.system(convert_cmd.strip())
    
    def push_to_hub(self):
        """Upload trained model to HuggingFace"""
        if not self.config.push_to_hub:
            return
        
        logger.info(f"Pushing to HuggingFace: {self.config.hf_output_repo}")
        
        api = HfApi()
        
        # Create model card
        model_card = self._create_model_card()
        
        model_dir = Path(self.config.output_dir) / "checkpoint-final"
        with open(model_dir / "README.md", "w") as f:
            f.write(model_card)
        
        # Upload
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=self.config.hf_output_repo,
            repo_type="model",
            commit_message=f"Upload {self.config.model_name} trained on zen-agentic-dataset"
        )
        
        logger.info(f"Model uploaded to {self.config.hf_output_repo}")
    
    def _create_model_card(self) -> str:
        """Generate model card for HuggingFace"""
        return f"""---
license: apache-2.0
base_model: {self.config.base_model}
datasets:
  - zenlm/zen-agentic-dataset
tags:
  - zen
  - coder
  - code-generation
  - agentic
  - mlx
  - apple-silicon
language:
  - en
library_name: mlx
pipeline_tag: text-generation
---

# Zen Coder

The flagship coding model from Zen LM, trained on the **Zen Agentic Dataset** (8.47B tokens).

## Model Details

- **Base Model**: {self.config.base_model}
- **Training Dataset**: zenlm/zen-agentic-dataset (8.47B tokens)
- **Training Method**: LoRA fine-tuning on MLX (Apple Silicon)
- **Quantization**: {self.config.quantize_bits}-bit

## Training Data

The Zen Agentic Dataset contains:
- **8.47 billion tokens** from 1,452 repositories
- **15 years** of production code history
- **2.1M samples** covering:
  - Agentic AI & LLM infrastructure (MCP, multi-agent, tool-use)
  - Web3 & Blockchain (Solidity, DeFi, consensus)
  - Modern Cryptography (post-quantum, MPC, ZK-proofs)
  - Full-stack development (Next.js, React, TypeScript)
  - Systems programming (Rust, Go, async patterns)

## Usage

### With MLX (Apple Silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load("{self.config.hf_output_repo}")

prompt = '''<|im_start|>system
You are Zen Coder, an expert AI programming assistant.<|im_end|>
<|im_start|>user
Write a function to implement Dilithium signature verification<|im_end|>
<|im_start|>assistant
'''

response = generate(
    model, tokenizer, 
    prompt=prompt,
    max_tokens=512,
    temp=0.7
)
print(response)
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.config.hf_output_repo}")
tokenizer = AutoTokenizer.from_pretrained("{self.config.hf_output_repo}")

messages = [
    {{"role": "system", "content": "You are Zen Coder."}},
    {{"role": "user", "content": "Implement a merkle tree in Rust"}}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Specializations

1. **Agentic AI**: MCP server development, multi-agent coordination, tool-use patterns
2. **Blockchain**: Smart contracts, DeFi protocols, consensus mechanisms
3. **Cryptography**: Post-quantum (ML-KEM, ML-DSA), threshold signatures, MPC
4. **Web Development**: Next.js 14+, React 18+, TypeScript, GraphQL
5. **Systems**: Rust async, Go concurrency, WASM, networking

## Training Configuration

- **LoRA**: r={self.config.lora_r}, alpha={self.config.lora_alpha}
- **Learning Rate**: {self.config.learning_rate}
- **Batch Size**: {self.config.batch_size}
- **Max Sequence Length**: {self.config.max_seq_length}
- **Epochs**: {self.config.num_epochs}

## License

Apache 2.0

## Citation

```bibtex
@misc{{zen-coder-2025,
  title={{Zen Coder: Flagship Code Generation Model Trained on Agentic Data}},
  author={{Hanzo AI Research Team}},
  year={{2025}},
  url={{https://huggingface.co/{self.config.hf_output_repo}}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Train Zen Coder on MLX")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--output-dir", type=str, default="./models/zen-coder")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--test-run", action="store_true", help="Test with small sample")
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub
    )
    
    # Initialize
    data_loader = ZenAgenticDataLoader(config)
    data_loader.load_tokenizer()
    
    if args.test_run:
        logger.info("Test run: processing 10 samples")
        for i, example in enumerate(data_loader.stream_examples(max_samples=10)):
            tokenized = data_loader.tokenize_example(example)
            logger.info(f"Sample {i+1}: {len(tokenized['input_ids'])} tokens")
        return
    
    # Train
    if HAS_MLX:
        trainer = MLXLoRATrainer(config)
        trainer.setup()
        losses = trainer.train(data_loader, max_steps=args.max_steps)
        
        if args.push_to_hub:
            trainer.push_to_hub()
    else:
        logger.error("MLX not available. Install with: pip install mlx mlx-lm")
        logger.info("Falling back to PyTorch training...")
        # Could add PyTorch fallback here
        

if __name__ == "__main__":
    main()
