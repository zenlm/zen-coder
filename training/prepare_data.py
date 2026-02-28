#!/usr/bin/env python3
"""
Prepare training data in mlx_lm format for LoRA fine-tuning.
Format: {"text": "conversation"} or {"messages": [...]}
"""

import json
from pathlib import Path

DATA_DIR = Path.home() / "work/zen/zen-agentic-dataset"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are Zen Coder, an expert AI programming assistant."""

def convert_debug_session(content: str) -> dict:
    """Convert a debug session to chat format"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Here is a coding session log. Learn from it:"},
        {"role": "assistant", "content": content[:8000]}  # Truncate long sessions
    ]
    return {"messages": messages}

def main():
    print("Preparing training data...")
    
    train_samples = []
    valid_samples = []
    
    # Load from debug sessions
    debug_file = DATA_DIR / "data/claude-debug/debug_sessions_latest.jsonl"
    if debug_file.exists():
        print(f"Processing {debug_file}...")
        with open(debug_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line.strip())
                    content = obj.get("content", "")
                    if content and len(content) > 500:
                        sample = convert_debug_session(content)
                        if i % 10 == 0:
                            valid_samples.append(sample)
                        else:
                            train_samples.append(sample)
                        
                        if len(train_samples) >= 1000:  # Limit for initial training
                            break
                except:
                    continue
    
    # Write output files
    train_file = OUTPUT_DIR / "train.jsonl"
    valid_file = OUTPUT_DIR / "valid.jsonl"
    
    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(valid_file, 'w') as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(train_samples)} training samples: {train_file}")
    print(f"Created {len(valid_samples)} validation samples: {valid_file}")
    print(f"\nTo train, run:")
    print(f"python -m mlx_lm.lora --model Qwen/Qwen3-4B-Instruct-2507 --train --data {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
