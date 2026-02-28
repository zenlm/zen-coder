"""
Zen Coder Unified Training Framework

Supports fine-tuning 5 model architectures:
  Tier 1: Qwen3 4B        (lightweight,   8GB VRAM)    $17-52
  Tier 2: Devstral 24B    (mid-range,    24GB VRAM)    $70-140
  Tier 3: Devstral 123B   (large,       128GB VRAM)   $140-280
  Tier 4: GLM-4.5 358B    (MoE,         128GB VRAM)   $210-420
  Tier 5: Kimi K2 1T      (MoE,         400GB VRAM)   $420-840

Costs based on 8xH200 @ $35/hr, 10K training samples.

Backends:
  - MLX (Apple Silicon, local) - Tier 1-2
  - Unsloth/PyTorch (NVIDIA GPUs, 2x faster) - Tier 1-4
  - DeepSpeed ZeRO-3 (Multi-GPU clusters) - All tiers
  - HuggingFace Autotrain (Cloud) - All tiers

Benchmarks (12 ARC from GLM-4.5):
  Agentic: TAU-Bench, BFCL V3, BrowseComp
  Reasoning: MMLU-Pro, AIME-24, MATH-500, SciCode, GPQA, HLE, LiveCodeBench
  Coding: SWE-bench Verified, Terminal-Bench

Usage:
    from zen_trainer import ZenTrainer, ZEN_MODELS, ZenBenchmark

    # Train
    trainer = ZenTrainer(
        model_key="qwen3-4b",
        dataset_path="hanzoai/zen-agentic-dataset",
        output_dir="./output/zen-coder-4b",
    )
    trainer.train()

    # Benchmark
    bench = ZenBenchmark(
        model_path="./output/zen-coder-4b",
        model_key="qwen3-4b",
    )
    bench.run_all()
    bench.compare_to_baseline()
"""

__version__ = "0.1.0"

from .models import ZEN_MODELS, get_model_config, estimate_training_cost, COST_SUMMARY
from .trainer import ZenTrainer, train_all_models
from .benchmark import ZenBenchmark, benchmark_all_models, generate_leaderboard

__all__ = [
    # Models
    "ZEN_MODELS",
    "get_model_config",
    "estimate_training_cost",
    "COST_SUMMARY",
    # Training
    "ZenTrainer",
    "train_all_models",
    # Benchmarks
    "ZenBenchmark",
    "benchmark_all_models",
    "generate_leaderboard",
]
