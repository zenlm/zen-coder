#!/usr/bin/env python3
"""
Zen Coder 4B Full Training Script
Trains on 30GB of agentic data with checkpointing and progress tracking.

Usage:
    python train_full.py                    # Start fresh
    python train_full.py --resume           # Resume from checkpoint
    python train_full.py --status           # Show training status
    python train_full.py --stop             # Gracefully stop training
"""

import os
import sys
import gc
import json
import glob
import time
import signal
import argparse
import resource
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any, Optional
import subprocess

# Set hard memory limit (20GB)
MAX_MEMORY_GB = 20
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_GB * 1024**3, hard))
    print(f"Memory limit set to {MAX_MEMORY_GB}GB")
except Exception as e:
    print(f"Could not set memory limit: {e}")

# MLX memory management
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Paths
DATA_DIR = Path("/Users/z/work/zen/zen-agentic-dataset/data")
TRAINING_DIR = Path("/Users/z/work/zen/zen-coder/training")
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
ADAPTER_DIR = TRAINING_DIR / "adapters"
STATUS_FILE = TRAINING_DIR / "training_status.json"
PREPARED_DATA = Path("/Users/z/work/zen/zen-coder")  # Main zen-coder directory
LOG_FILE = TRAINING_DIR / "full_training.log"

# Training config
BASE_MODEL = "Qwen/Qwen3-Coder-Next"
BATCH_SIZE = 1  # Small batch, use grad accumulation
GRAD_ACCUMULATION = 4  # Effective batch = 4
LEARNING_RATE = 1e-5  # Conservative LR
LORA_LAYERS = 4  # Minimal layers to fit in 64GB RAM
MAX_SEQ_LENGTH = 1024  # Reduced to fit in 64GB RAM
MAX_CHARS_PER_CHUNK = 12000  # ~3K tokens per chunk
CHECKPOINT_EVERY = 5000  # Checkpoint every 5K iters for long training
EVAL_EVERY = 25000  # Eval every 25K to save time
MAX_ITERATIONS = 837500  # 1 full epoch over 3.35M samples at batch=4
FINE_TUNE_TYPE = "lora"  # or "dora" for DoRA
OPTIMIZER = "adamw"  # AdamW with weight decay
KEEP_CHECKPOINTS = 5  # Keep more checkpoints for long training

# Identity data path (prepended to training data)
IDENTITY_DATA = Path("/Users/z/work/zen/zen-identity-dataset")

# Graceful shutdown flag
shutdown_requested = False


def cleanup_old_checkpoints():
    """Delete old checkpoints, keeping only the most recent KEEP_CHECKPOINTS."""
    if not ADAPTER_DIR.exists():
        return

    # Find all numbered checkpoint files
    checkpoints = sorted(ADAPTER_DIR.glob("*_adapters.safetensors"))

    if len(checkpoints) <= KEEP_CHECKPOINTS:
        return

    # Delete oldest checkpoints
    to_delete = checkpoints[:-KEEP_CHECKPOINTS]
    for ckpt in to_delete:
        try:
            ckpt.unlink()
            print(f"  Deleted old checkpoint: {ckpt.name}")
        except Exception as e:
            print(f"  Warning: Could not delete {ckpt.name}: {e}")


def cleanup_memory():
    """Aggressive memory cleanup to prevent leaks."""
    gc.collect()
    if HAS_MLX:
        try:
            mx.metal.clear_cache()
        except (AttributeError, Exception):
            pass  # Not all MLX versions have this

def signal_handler(sig, frame):
    global shutdown_requested
    log("\n⚠️  Shutdown requested - will save checkpoint and exit...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_status() -> Dict[str, Any]:
    """Load training status from file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {
        "started": None,
        "iteration": 0,
        "total_iterations": 0,
        "samples_processed": 0,
        "total_samples": 0,
        "loss": None,
        "eta": None,
        "status": "not_started",
        "last_checkpoint": None,
        "cumulative_iterations": 0,  # Track across restarts
    }


def save_status(status: Dict[str, Any]):
    """Save training status to file."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2, default=str)


def count_jsonl_lines(path: Path) -> int:
    """Count lines in JSONL file efficiently."""
    count = 0
    with open(path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


def discover_data_files() -> list[Path]:
    """Discover all JSONL files in data directory."""
    files = []
    for pattern in ["**/*.jsonl"]:
        files.extend(DATA_DIR.glob(pattern))
    # Sort by size (largest first for better estimates)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def chunk_messages(messages: list, max_chars: int = MAX_CHARS_PER_CHUNK) -> list:
    """Split long message sequences into chunks that fit within token limits.

    Optimized for large content (up to 100M+ chars) with O(n) complexity.
    """
    chunks = []
    current_chunk = []
    current_len = 0

    for msg in messages:
        content = msg.get("content", "")
        msg_len = len(content)
        role = msg.get("role", "assistant")

        # If single message exceeds limit, split it
        if msg_len > max_chars:
            # For very large content (>100K), use simple hard splits for O(n) performance
            # Trying to find paragraph boundaries in 4M+ char strings is too slow
            if msg_len > 100000:
                # Hard split on chunk boundaries - O(n) linear time
                pos = 0
                while pos < msg_len:
                    end = min(pos + max_chars, msg_len)
                    # Try to break at newline if within last 1000 chars
                    if end < msg_len:
                        last_newline = content.rfind('\n', pos + max_chars - 1000, end)
                        if last_newline > pos:
                            end = last_newline + 1
                    chunk_content = content[pos:end].strip()
                    if chunk_content:
                        chunks.append([{"role": role, "content": chunk_content}])
                    pos = end
            else:
                # For moderately large content, try paragraph boundaries
                # Use list append + join for O(n) instead of string += O(n²)
                parts = content.split("\n\n")
                buffer_parts = []
                buffer_len = 0

                for part in parts:
                    part_len = len(part)
                    # Would adding this part exceed limit?
                    if buffer_len + part_len + 2 > max_chars:  # +2 for \n\n
                        if buffer_parts:
                            chunk_content = "\n\n".join(buffer_parts).strip()
                            if chunk_content:
                                if current_chunk:
                                    current_chunk.append({"role": role, "content": chunk_content})
                                    chunks.append(current_chunk)
                                    current_chunk = []
                                else:
                                    chunks.append([{"role": role, "content": chunk_content}])
                            buffer_parts = []
                            buffer_len = 0
                        # If single part is still too long, hard split
                        while len(part) > max_chars:
                            chunks.append([{"role": role, "content": part[:max_chars]}])
                            part = part[max_chars:]
                        if part:
                            buffer_parts = [part]
                            buffer_len = len(part)
                    else:
                        buffer_parts.append(part)
                        buffer_len += part_len + 2

                if buffer_parts:
                    chunk_content = "\n\n".join(buffer_parts).strip()
                    if chunk_content:
                        current_chunk.append({"role": role, "content": chunk_content})
                        current_len = len(chunk_content)
        elif current_len + msg_len > max_chars:
            # Start new chunk
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [msg]
            current_len = msg_len
        else:
            current_chunk.append(msg)
            current_len += msg_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [messages]


def prepare_data_for_mlx():
    """Convert all JSONL data to mlx_lm format with chunking for long samples."""
    log("📊 Discovering data files...")
    files = discover_data_files()

    total_size = sum(f.stat().st_size for f in files)
    log(f"Found {len(files)} files, {total_size / 1e9:.2f} GB total")

    PREPARED_DATA.mkdir(parents=True, exist_ok=True)
    train_file = PREPARED_DATA / "train.jsonl"
    valid_file = PREPARED_DATA / "valid.jsonl"

    total_samples = 0
    valid_samples = 0
    identity_samples = 0

    system_prompt = """You are Zen Coder, an expert AI coding assistant created by Zen LM and Hanzo AI.
You excel at writing clean, efficient code across multiple languages including Python, TypeScript, Rust, and Go.
You help with debugging, code review, architecture design, and explaining complex technical concepts."""

    log("📝 Converting to mlx_lm format...")

    # First, add identity data at the beginning (high priority)
    identity_files = list(IDENTITY_DATA.glob("*.jsonl")) if IDENTITY_DATA.exists() else []
    if identity_files:
        log(f"🆔 Adding identity data from {len(identity_files)} files...")
        with open(train_file, 'w') as train_f:
            for id_file in identity_files:
                with open(id_file, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if "messages" in item:
                                # Ensure system prompt
                                msgs = item["messages"]
                                if msgs and msgs[0].get("role") != "system":
                                    msgs.insert(0, {"role": "system", "content": system_prompt})
                                train_f.write(json.dumps({"messages": msgs}) + "\n")
                                identity_samples += 1
                        except:
                            continue
        log(f"   Added {identity_samples:,} identity samples")
        total_samples = identity_samples
        # Open in append mode for rest of data
        train_mode = 'a'
    else:
        log("⚠️  No identity data found, starting with main dataset")
        train_mode = 'w'

    with open(train_file, train_mode) as train_f, open(valid_file, 'w') as valid_f:
        for i, data_file in enumerate(files):
            log(f"  [{i+1}/{len(files)}] Processing {data_file.name} ({data_file.stat().st_size / 1e6:.1f} MB)...")
            
            try:
                file_lines = 0
                file_samples = 0
                with open(data_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        file_lines += 1
                        # Progress every 10K lines
                        if file_lines % 10000 == 0:
                            print(f"    ... {file_lines:,} lines, {file_samples:,} samples, {total_samples:,} total", flush=True)
                        try:
                            item = json.loads(line)
                            
                            # Convert to messages format
                            messages = []
                            
                            # Handle different formats
                            if "messages" in item:
                                messages = item["messages"]
                            elif "conversations" in item:
                                for conv in item["conversations"]:
                                    role = "assistant" if conv.get("from") == "gpt" else "user"
                                    messages.append({"role": role, "content": conv.get("value", "")})
                            elif "prompt" in item and "completion" in item:
                                messages = [
                                    {"role": "user", "content": item["prompt"]},
                                    {"role": "assistant", "content": item["completion"]}
                                ]
                            elif "text" in item:
                                # Single text - treat as assistant response (NO truncation - chunking handles it)
                                messages = [
                                    {"role": "user", "content": "Continue the following:"},
                                    {"role": "assistant", "content": item["text"]}
                                ]
                            elif "message" in item and item.get("type") in ("user", "assistant"):
                                # Claude conversation format with message field
                                content = item.get("message", "")
                                if isinstance(content, dict):
                                    content = content.get("content", str(content))
                                role = item.get("type", "user")
                                if role == "human":
                                    role = "user"
                                messages = [{"role": role, "content": str(content)}]
                            elif "content" in item:
                                # Git history or debug session format - FULL content, chunking handles splitting
                                content = item.get("content", "")
                                if isinstance(content, str) and len(content) > 50:
                                    item_type = item.get("type", "code")
                                    if item_type in ("commit", "diff", "file"):
                                        # Code training
                                        filename = item.get("filename", "file")
                                        messages = [
                                            {"role": "user", "content": f"Analyze this code from {filename}:"},
                                            {"role": "assistant", "content": content}  # Full content
                                        ]
                                    else:
                                        # Session content - FULL session, chunking splits it
                                        messages = [
                                            {"role": "user", "content": "Continue this development session:"},
                                            {"role": "assistant", "content": content}  # Full content
                                        ]
                            else:
                                continue  # Skip unknown format
                            
                            # Skip if too short
                            total_content = sum(len(m.get("content", "")) for m in messages)
                            if total_content < 100:
                                continue
                            
                            # Chunk long sequences
                            message_chunks = chunk_messages(messages, MAX_CHARS_PER_CHUNK)
                            
                            for chunk in message_chunks:
                                # Add system prompt to each chunk
                                if chunk and chunk[0].get("role") != "system":
                                    chunk.insert(0, {"role": "system", "content": system_prompt})

                                output = json.dumps({"messages": chunk})
                                file_samples += 1

                                # 95% train, 5% validation
                                if line_num % 20 == 0:
                                    valid_f.write(output + "\n")
                                    valid_samples += 1
                                else:
                                    train_f.write(output + "\n")
                                    total_samples += 1
                                
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            continue  # Skip malformed lines

                log(f"    ✓ {file_lines:,} lines → {file_samples:,} samples")

            except Exception as e:
                log(f"    ⚠️  Error processing {data_file.name}: {e}")
                continue
    
    log(f"✅ Prepared {total_samples:,} training samples, {valid_samples:,} validation samples")
    return total_samples, valid_samples


def estimate_training_time(total_samples: int, batch_size: int = BATCH_SIZE) -> str:
    """Estimate training time based on sample count."""
    # Rough estimate: ~0.5 seconds per sample on M-series Mac with 4B model
    total_iterations = total_samples // batch_size
    seconds_per_iter = 2.0  # Conservative estimate
    total_seconds = total_iterations * seconds_per_iter
    
    if total_seconds < 3600:
        return f"{total_seconds / 60:.0f} minutes"
    elif total_seconds < 86400:
        return f"{total_seconds / 3600:.1f} hours"
    else:
        return f"{total_seconds / 86400:.1f} days"


def run_training(resume: bool = False):
    """Run the full training with mlx_lm."""
    global shutdown_requested
    
    status = load_status()
    
    # Prepare data if needed
    train_file = PREPARED_DATA / "train.jsonl"
    if not train_file.exists() or train_file.stat().st_size < 1000:
        total_samples, valid_samples = prepare_data_for_mlx()
        status["total_samples"] = total_samples
        save_status(status)
    else:
        # Count existing samples
        total_samples = count_jsonl_lines(train_file)
        log(f"📂 Using existing prepared data: {total_samples:,} samples")
    
    # Track cumulative iterations across restarts
    cumulative = status.get("cumulative_iterations", 0)
    target_iterations = min(MAX_ITERATIONS, total_samples // BATCH_SIZE)
    remaining_iterations = max(0, target_iterations - cumulative)
    
    if remaining_iterations == 0:
        log(f"✅ Training already complete! {cumulative:,} total iterations done.")
        return
    
    total_iterations = remaining_iterations
    estimated_time = f"{total_iterations * 5 / 3600:.1f} hours"  # ~5 sec/iter
    log(f"   Cumulative progress: {cumulative:,} / {target_iterations:,} iterations")
    
    log(f"\n🚀 Starting Zen Coder 4B Training")
    log(f"   Model: {BASE_MODEL}")
    log(f"   Samples: {total_samples:,}")
    log(f"   Iterations: {total_iterations:,}")
    log(f"   Effective batch: {BATCH_SIZE} x {GRAD_ACCUMULATION} = {BATCH_SIZE * GRAD_ACCUMULATION}")
    log(f"   LoRA layers: {LORA_LAYERS} {'(ALL)' if LORA_LAYERS == -1 else ''}, type: {FINE_TUNE_TYPE}")
    log(f"   Max seq length: {MAX_SEQ_LENGTH}")
    log(f"   Estimated time: {estimated_time}")
    log(f"\n   Press Ctrl+C to gracefully stop and save checkpoint\n")
    
    # Update status
    status["started"] = datetime.now().isoformat()
    status["total_samples"] = total_samples
    status["total_iterations"] = total_iterations
    status["status"] = "running"
    save_status(status)
    
    # Build mlx_lm command with best practices
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", str(PREPARED_DATA),
        "--fine-tune-type", FINE_TUNE_TYPE,
        "--optimizer", OPTIMIZER,
        "--iters", str(total_iterations),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accumulation-steps", str(GRAD_ACCUMULATION),
        "--num-layers", str(LORA_LAYERS),
        "--adapter-path", str(ADAPTER_DIR),
        "--learning-rate", str(LEARNING_RATE),
        "--save-every", str(CHECKPOINT_EVERY),
        "--steps-per-eval", str(EVAL_EVERY),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--mask-prompt",  # Train on completions only
        "--grad-checkpoint",  # Save memory with gradient checkpointing
    ]
    
    if resume and (ADAPTER_DIR / "adapters.safetensors").exists():
        cmd.extend(["--resume-adapter-file", str(ADAPTER_DIR / "adapters.safetensors")])
        log("📥 Resuming from checkpoint...")
    
    # Create log directory
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    log(f"📝 Logging to: {LOG_FILE}")
    log("─" * 60)
    
    start_time = time.time()
    last_update = 0
    
    with open(LOG_FILE, 'a') as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Training started: {datetime.now().isoformat()}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"{'='*60}\n\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(TRAINING_DIR),
        )
        
        try:
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
                
                # Parse progress
                if "Iter" in line:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "Iter":
                                iter_num = int(parts[i+1].rstrip(':'))
                                status["iteration"] = iter_num
                            if "loss" in p.lower() and i + 1 < len(parts):
                                try:
                                    loss = float(parts[i+1].rstrip(','))
                                    status["loss"] = loss
                                except:
                                    pass
                    except:
                        pass
                    
                    # Update ETA
                    elapsed = time.time() - start_time
                    if status["iteration"] > 0:
                        rate = elapsed / status["iteration"]
                        remaining = (total_iterations - status["iteration"]) * rate
                        eta = datetime.now() + timedelta(seconds=remaining)
                        status["eta"] = eta.isoformat()
                        
                        # Show progress every 10 iterations
                        if status["iteration"] - last_update >= 10:
                            last_update = status["iteration"]
                            pct = 100 * status["iteration"] / total_iterations
                            log(f"  Iter {status['iteration']:,}/{total_iterations:,} ({pct:.1f}%) | "
                                  f"Loss: {status.get('loss', 'N/A'):.4f} | "
                                  f"ETA: {timedelta(seconds=int(remaining))}")

                        # Cleanup old checkpoints after each checkpoint save
                        if status["iteration"] % CHECKPOINT_EVERY == 0:
                            cleanup_old_checkpoints()
                        
                        # Aggressive memory cleanup every 100 iterations
                        if status["iteration"] % 100 == 0:
                            cleanup_memory()

                    save_status(status)
                
                if shutdown_requested:
                    log("\n⚠️  Saving checkpoint and stopping...")
                    process.terminate()
                    process.wait(timeout=30)
                    break
                    
        except KeyboardInterrupt:
            log("\n⚠️  Interrupted - saving checkpoint...")
            process.terminate()
            process.wait(timeout=30)
        finally:
            # Update cumulative iterations across restarts
            session_iters = status.get("iteration", 0)
            cumulative = status.get("cumulative_iterations", 0) + session_iters
            status["cumulative_iterations"] = cumulative
            status["status"] = "stopped" if shutdown_requested else "completed"
            save_status(status)
            log(f"  Cumulative iterations: {cumulative:,}")
            
            elapsed = time.time() - start_time
            log(f"\n{'='*60}")
            log(f"Training {'stopped' if shutdown_requested else 'completed'}!")
            log(f"  Iterations: {status['iteration']:,}/{total_iterations:,}")
            log(f"  Time: {timedelta(seconds=int(elapsed))}")
            log(f"  Final loss: {status.get('loss', 'N/A')}")
            log(f"  Adapter saved to: {ADAPTER_DIR}")
            log(f"\nTo resume: python train_full.py --resume")


def show_status():
    """Show current training status."""
    status = load_status()
    
    log("\n📊 Zen Coder Training Status")
    log("─" * 40)
    log(f"  Status: {status['status']}")
    log(f"  Started: {status.get('started', 'N/A')}")
    log(f"  Progress: {status.get('iteration', 0):,} / {status.get('total_iterations', 0):,} iterations")
    if status.get('total_iterations', 0) > 0:
        pct = 100 * status.get('iteration', 0) / status['total_iterations']
        log(f"  Percent: {pct:.1f}%")
    log(f"  Samples: {status.get('samples_processed', 0):,} / {status.get('total_samples', 0):,}")
    log(f"  Loss: {status.get('loss', 'N/A')}")
    log(f"  ETA: {status.get('eta', 'N/A')}")
    log(f"  Last checkpoint: {status.get('last_checkpoint', 'N/A')}")
    
    # Check if adapter exists
    adapter_file = ADAPTER_DIR / "adapters.safetensors"
    if adapter_file.exists():
        size_mb = adapter_file.stat().st_size / 1e6
        log(f"\n📦 Adapter: {adapter_file} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Zen Coder 4B Full Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--stop", action="store_true", help="Request graceful stop")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, don't train")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild training data from full dataset")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.stop:
        status = load_status()
        status["status"] = "stop_requested"
        save_status(status)
        log("⚠️  Stop requested. Training will save checkpoint and exit.")
        return

    if args.rebuild:
        log("🔄 Forcing rebuild of training data...")
        train_file = PREPARED_DATA / "train.jsonl"
        valid_file = PREPARED_DATA / "valid.jsonl"
        if train_file.exists():
            train_file.unlink()
        if valid_file.exists():
            valid_file.unlink()
        prepare_data_for_mlx()
        if not args.prepare_only:
            run_training(resume=False)
        return

    if args.prepare_only:
        prepare_data_for_mlx()
        return

    run_training(resume=args.resume)


if __name__ == "__main__":
    main()
