# Zen Coder Model Card

**Family of agentic coding models fine-tuned on 8.47B tokens of real-world AI programming data.**

## Model Variants

| Model | Base | Size | VRAM | Context | License |
|-------|------|------|------|---------|---------|
| **Zen Coder 4B** | Qwen3-4B-Instruct | 4B | 8 GB | 32K | Apache 2.0 |
| **Zen Coder 24B** | Devstral-Small-2-24B | 24B | 24 GB | 256K | Apache 2.0 |
| **Zen Coder 123B** | Devstral-2-123B | 123B | 128 GB | 256K | Mistral Research |
| **Zen Coder MAX** | GLM-4.7 | 358B (MoE) | 180 GB | 200K | GLM-4 License |

## Intended Use

Zen Coder models are designed for:

- **Agentic coding** - Multi-step programming tasks with tool use
- **Code generation** - Functions, classes, full files across 50+ languages
- **Code understanding** - Explanation, refactoring, debugging
- **Architecture design** - System design, API design, technical planning
- **DevOps automation** - CI/CD, deployment, infrastructure as code

### Primary Languages
Python, TypeScript, JavaScript, Rust, Go, Solidity, SQL, Bash

### Specialized Domains
- AI/ML infrastructure (MCP, agents, LLM orchestration)
- Blockchain/Web3 (smart contracts, consensus, DeFi)
- Cryptography (post-quantum, threshold, ZKP)
- Full-stack web development (Next.js, React, Node.js)

## Training Data

**Dataset:** [Zen Agentic Dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset) (private)

| Metric | Value |
|--------|-------|
| Total Tokens | 8.47 billion |
| Training Samples | 3.35 million |
| Validation Samples | 100,000 |
| Total Size | 27 GB |
| Time Span | 15 years (2010-2025) |
| Repositories | 1,452 |

### Data Composition
- **29%** Claude Code debug sessions (real agentic programming)
- **23%** Claude conversations and interactions
- **48%** Git history (commits, diffs, source files)

## Training Details

### Framework
- **Local:** MLX (Apple Silicon)
- **Cloud:** Unsloth + DeepSpeed (8xH200)
- **Config:** QLoRA with gradient checkpointing

### Hyperparameters

| Model | LoRA r | LoRA α | Batch | LR | Epochs |
|-------|--------|--------|-------|-----|--------|
| 4B | 64 | 128 | 4 | 2e-4 | 2 |
| 24B | 32 | 64 | 2 | 1e-4 | 2 |
| 123B | 16 | 32 | 1 | 5e-5 | 1 |
| MAX | 16 | 32 | 1 | 5e-6 | 1 |

### Training Costs (3.35M samples, 8xH200 @ $35/hr)

| Model | Hours | Cost | Local (Mac Studio) |
|-------|-------|------|-------------------|
| Zen Coder 4B | 9h | $326 | 2 days (FREE) |
| Zen Coder 24B | 23h | $814 | 5 days (FREE) |
| Zen Coder 123B | 62h | $2,171 | 13 days (FREE) |
| Zen Coder MAX | 116h | $4,071 | 19 days (FREE) |

## Evaluation

### Benchmarks (12 ARC from GLM-4.5)

**Agentic:**
- TAU-Bench (tool-agent-user interaction)
- BFCL V3 (Berkeley Function Call Leaderboard)
- BrowseComp (web browsing agent)

**Reasoning:**
- MMLU-Pro, AIME-24, MATH-500, SciCode
- GPQA, HLE (Humanity's Last Exam)
- LiveCodeBench

**Coding:**
- SWE-bench Verified (real GitHub issues)
- Terminal-Bench (terminal environment tasks)

Evaluation toolkit: https://github.com/zai-org/glm-simple-evals

### Expected Performance

Fine-tuning on domain-specific agentic data is expected to improve:
- Tool use accuracy (function calling, MCP)
- Multi-step reasoning (agentic workflows)
- Code quality in specialized domains (Web3, AI infra, crypto)
- Context utilization for long files and conversations

## Limitations

- **Not general-purpose:** Optimized for coding, may underperform on non-technical tasks
- **Domain bias:** Strong in AI/Web3/crypto, may be weaker in other domains
- **English-centric:** Training data primarily in English
- **Code style:** May reflect patterns from training repositories

## Ethical Considerations

- Training data derived from open-source repositories and personal Claude sessions
- No PII or secrets in training data (filtered)
- Models may generate code with security vulnerabilities (always review)
- Not intended for generating malicious code

## Usage

```python
from zen_trainer import ZenTrainer, ZenBenchmark

# Train
trainer = ZenTrainer(
    model_key="qwen3-4b",  # or devstral-24b, devstral-123b, glm47-358b
    dataset_path="hanzoai/zen-agentic-dataset",
    output_dir="./output/zen-coder-4b",
)
trainer.train()

# Benchmark
bench = ZenBenchmark(
    model_path="./output/zen-coder-4b",
    model_key="qwen3-4b",
)
results = bench.run_all()
bench.compare_to_baseline()
```

## Model Access

| Model | Status | HuggingFace |
|-------|--------|-------------|
| Zen Coder 4B | Training | `zenlm/zen-coder-4b` |
| Zen Coder 24B | Planned | `zenlm/zen-coder-24b` |
| Zen Coder 123B | Planned | `zenlm/zen-coder-123b` |
| Zen Coder MAX | Planned | `zenlm/zen-coder-max` |

## Citation

```bibtex
@model{zen_coder,
  author = {Kelling, Zach},
  title = {Zen Coder: Agentic Coding Models Fine-tuned on 8.47B Tokens},
  year = {2025},
  publisher = {Zoo Labs Foundation},
  note = {4 model variants (4B-358B), trained on zen-agentic-dataset},
  url = {https://github.com/zenlm/zen-coder}
}
```

## Related Projects

- [Zen Agentic Dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset) - Training data
- [Hanzo MCP](https://github.com/hanzoai/mcp) - Model Context Protocol (260+ tools)
- [Hanzo AI](https://hanzo.ai) - AI infrastructure platform
- [Zoo Labs](https://zoo.ngo) - Decentralized AI research

---

**Version:** 0.1.0
**Last Updated:** 2025-12-30
**Maintainer:** z@hanzo.ai
