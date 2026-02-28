#!/usr/bin/env python3
"""
Quick MLX training using local data files.
"""

import json
import logging
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local data paths
DATA_DIR = Path.home() / "work/zen/zen-agentic-dataset"
LOCAL_FILES = [
    DATA_DIR / "data/claude-debug/debug_sessions_latest.jsonl",
]

def stream_local_data(max_samples=1000):
    """Stream data from local JSONL files"""
    count = 0
    for file_path in LOCAL_FILES:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"Processing: {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                if count >= max_samples:
                    return
                try:
                    obj = json.loads(line.strip())
                    content = obj.get("content", "")
                    if content and len(content) > 100:
                        yield content
                        count += 1
                except:
                    continue
    logger.info(f"Loaded {count} samples")

def main():
    logger.info("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-4B-Instruct-2507")
    
    logger.info("Testing inference...")
    prompt = "Write a Python function to calculate fibonacci:"
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
    print(f"\nTest output:\n{response}\n")
    
    logger.info("Loading local data...")
    samples = list(stream_local_data(max_samples=100))
    logger.info(f"Loaded {len(samples)} samples for training test")
    
    # Basic inference test on sample
    if samples:
        sample = samples[0][:500]  # First 500 chars
        logger.info(f"Sample preview: {sample[:200]}...")
    
    logger.info("Training setup complete!")
    logger.info("For full MLX training, use: python -m mlx_lm.lora --model Qwen/Qwen3-4B-Instruct-2507 --data ./data")

if __name__ == "__main__":
    main()
