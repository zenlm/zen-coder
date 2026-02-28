# zen-coder — AI Knowledge Base

**Project**: zen-coder
**Organization**: zenlm
**Repository**: https://github.com/zenlm/zen-coder
**HuggingFace**: https://huggingface.co/zenlm/zen-coder
**Last Updated**: 2026-02-27

## Overview

zen-coder is the code-focused model family from Zen LM. Models range from 4B (edge) to 480B (MoE frontier).
Built on Qwen3-Coder architecture with extended context (128K tokens).

## Model Variants

| Model | Params | Base | Context |
|-------|--------|------|---------|
| zen-coder (4B) | 4B | Qwen3-Coder-4B | 32K |
| zen-coder-flash (31B MoE) | 31B/3B active | GLM-4.7-Flash | 131K |
| zen-coder-480b | 480B/30B active | Qwen3-Coder | 128K |

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit model weights (*.safetensors, *.bin, *.gguf, *.pt)
3. **NEVER** commit symlinked files (CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md)
4. **NEVER** create random summary files — update THIS file only
5. Zen models are based on **Qwen3** (not Qwen2!)

## Context

This file (`LLM.md`) is symlinked as CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md.

---

*Part of the Zen AI family — Clarity Through Intelligence*
