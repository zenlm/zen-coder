#!/usr/bin/env python3
"""
Zen Model Server - Unified MLX inference with hot-swap
=====================================================

Serves all Zen models (coder, flash, vision) with OpenAI-compatible API.
Supports hot-swapping between models and vision inputs.

Usage:
    python serve.py                          # Start with zen-coder (default)
    python serve.py --model zen-coder-flash  # Start with flash model
    python serve.py --port 3690              # Custom port

API:
    POST /v1/chat/completions    - Chat completions (OpenAI compatible)
    POST /v1/completions         - Text completions
    GET  /v1/models              - List available models
    POST /v1/models/load         - Hot-swap to a different model
    GET  /health                 - Health check
"""

import os
import gc
import sys
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm import load, generate
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zen")

# ── Model Registry ──────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    id: str
    name: str
    path: str
    family: str  # "coder", "flash", "vl"
    context_length: int
    vision: bool = False
    description: str = ""

MODELS: Dict[str, ModelSpec] = {
    "zen-coder": ModelSpec(
        id="zen-coder",
        name="Zen Coder",
        path="lmstudio-community/Qwen3-Coder-Next-MLX-4bit",
        family="coder",
        context_length=131072,
        description="Flagship coding model based on Qwen3-Coder-Next",
    ),
    "zen-coder-flash": ModelSpec(
        id="zen-coder-flash",
        name="Zen Coder Flash",
        path="lmstudio-community/GLM-4.7-Flash-MLX-6bit",
        family="flash",
        context_length=131072,
        description="Fast coding model based on GLM-4.7-Flash (31B MoE, 3B active)",
    ),
    "zen-vl-4b": ModelSpec(
        id="zen-vl-4b",
        name="Zen VL 4B",
        path="lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        family="vl",
        context_length=32768,
        vision=True,
        description="Vision-language model (4B) for multimodal tasks",
    ),
    "zen-vl-8b": ModelSpec(
        id="zen-vl-8b",
        name="Zen VL 8B",
        path="lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
        family="vl",
        context_length=32768,
        vision=True,
        description="Vision-language model (8B) for multimodal tasks",
    ),
    "zen-vl-30b": ModelSpec(
        id="zen-vl-30b",
        name="Zen VL 30B",
        path="lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit",
        family="vl",
        context_length=32768,
        vision=True,
        description="Vision-language model (30B MoE) for multimodal tasks",
    ),
}

# ── Request/Response Types ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts (for vision)

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

class LoadModelRequest(BaseModel):
    model: str

# ── Model Manager ───────────────────────────────────────────────────────────

class ZenModelManager:
    """Manages model loading/unloading with memory-efficient hot-swap."""

    def __init__(self):
        self.current_id: Optional[str] = None
        self.model = None
        self.tokenizer = None

    def load(self, model_id: str):
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")

        if model_id == self.current_id:
            log.info(f"Model {model_id} already loaded")
            return

        # Unload current model to free memory
        if self.model is not None:
            log.info(f"Unloading {self.current_id}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            mx.metal.reset_peak_memory()

        spec = MODELS[model_id]
        log.info(f"Loading {model_id} from {spec.path}...")

        start = time.time()
        self.model, self.tokenizer = load(spec.path)
        elapsed = time.time() - start

        self.current_id = model_id
        log.info(f"Loaded {model_id} in {elapsed:.1f}s")

    def generate_text(self, prompt: str, max_tokens: int = 2048,
                      temperature: float = 0.7, top_p: float = 0.95) -> str:
        if self.model is None:
            raise RuntimeError("No model loaded")

        return generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )

    @property
    def spec(self) -> Optional[ModelSpec]:
        return MODELS.get(self.current_id) if self.current_id else None

# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Zen Model Server", version="1.0.0")
mgr = ZenModelManager()

@app.on_event("startup")
async def startup():
    default = os.environ.get("ZEN_DEFAULT_MODEL", "zen-coder")
    mgr.load(default)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": mgr.current_id,
        "family": mgr.spec.family if mgr.spec else None,
        "vision": mgr.spec.vision if mgr.spec else False,
        "available": list(MODELS.keys()),
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": spec.id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "zenlm",
                "name": spec.name,
                "description": spec.description,
                "family": spec.family,
                "vision": spec.vision,
                "context_length": spec.context_length,
                "loaded": spec.id == mgr.current_id,
            }
            for spec in MODELS.values()
        ],
    }

@app.post("/v1/models/load")
async def load_model(req: LoadModelRequest):
    try:
        mgr.load(req.model)
        return {"status": "ok", "model": mgr.current_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def _format_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    parts = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            # Vision content - extract text parts
            text_parts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts) if text_parts else str(content)
        parts.append(f"<|im_start|>{msg.role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Auto-swap model if specified
    if req.model and req.model != mgr.current_id and req.model in MODELS:
        mgr.load(req.model)

    prompt = _format_messages(req.messages)
    response = mgr.generate_text(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    return {
        "id": f"chatcmpl-zen-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": mgr.current_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split()),
        },
    }

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if req.model and req.model != mgr.current_id and req.model in MODELS:
        mgr.load(req.model)

    response = mgr.generate_text(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    return {
        "id": f"cmpl-zen-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": mgr.current_id,
        "choices": [{
            "text": response,
            "index": 0,
            "finish_reason": "stop",
        }],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zen Model Server")
    parser.add_argument("--model", default="zen-coder", choices=list(MODELS.keys()))
    parser.add_argument("--port", type=int, default=3690)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.environ["ZEN_DEFAULT_MODEL"] = args.model
    print(f"Starting Zen Model Server on {args.host}:{args.port}")
    print(f"Default model: {args.model}")
    print(f"Available models: {', '.join(MODELS.keys())}")
    uvicorn.run(app, host=args.host, port=args.port)
