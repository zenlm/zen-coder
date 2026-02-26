# Zen Coder

Code-focused language model family from [Zen LM](https://zenlm.org).

Zen Coder is a family of models purpose-built for software engineering tasks: code generation, completion, debugging, refactoring, and agentic coding workflows.

## Model Variants

| Model | Parameters | Context | Use Case |
|-------|-----------|---------|----------|
| **zen-coder** | 4B | 128K | Edge / local development |
| **zen-coder-flash** | 31B MoE (3B active) | 131K | Balanced performance |
| **zen-coder-max** | 671B MoE (14B active) | 128K | Frontier coding |

## Features

- 128K context window for large codebases
- 92+ programming languages
- Fill-in-the-middle (FIM) completion
- Native function calling and tool use
- Strong math and reasoning alongside code

## Special Tokens

```json
{
  "<|fim_prefix|>": 151659,
  "<|fim_middle|>": 151660,
  "<|fim_suffix|>": 151661,
  "<|fim_pad|>": 151662,
  "<|repo_name|>": 151663,
  "<|file_sep|>": 151664,
  "<|im_start|>": 151644,
  "<|im_end|>": 151645
}
```

## Quick Start

### Requirements

```
python>=3.9
transformers>=4.37.0
```

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen-coder"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "Write a binary search in Python."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Fill-in-the-Middle

```python
prompt = "<|fim_prefix|>def fibonacci(n):\n    <|fim_suffix|>\n    return result<|fim_middle|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### vLLM (Production)

```bash
pip install vllm
vllm serve zenlm/zen-coder --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="zen")
response = client.chat.completions.create(
    model="zenlm/zen-coder",
    messages=[{"role": "user", "content": "Implement a binary heap in Go."}],
)
print(response.choices[0].message.content)
```

### Ollama

```bash
ollama run zenlm/zen-coder
```

## Quantized Formats

| Format | Size | Download |
|--------|------|----------|
| SafeTensors (BF16) | ~8 GB | [HuggingFace](https://huggingface.co/zenlm/zen-coder) |
| GGUF Q4_K_M | ~2.5 GB | [HuggingFace](https://huggingface.co/zenlm/zen-coder) |
| GGUF Q8_0 | ~4.5 GB | [HuggingFace](https://huggingface.co/zenlm/zen-coder) |
| AWQ Int4 | ~2.5 GB | [HuggingFace](https://huggingface.co/zenlm/zen-coder) |
| GPTQ Int4 | ~2.5 GB | [HuggingFace](https://huggingface.co/zenlm/zen-coder) |

## Supported Languages

Ada, Agda, Assembly, Awk, Bash, C, C#, C++, Clojure, CMake, CoffeeScript, CSS, CUDA, Dart,
Dockerfile, Elixir, Elm, Erlang, F#, Fortran, GLSL, Go, Groovy, Haskell, HTML, Java, JavaScript,
JSON, Julia, Jupyter, Kotlin, Lean, Lua, Makefile, Markdown, MATLAB, OCaml, Pascal, Perl, PHP,
PowerShell, Prolog, Python, R, Racket, Ruby, Rust, Scala, Scheme, Solidity, SQL, Swift,
SystemVerilog, TCL, TypeScript, VHDL, Vue, YAML, Zig, and 30+ more.

## Agentic Use

Zen Coder is optimized for use with [Zen Agent](https://github.com/zenlm/zen-agent) for multi-step
coding tasks, repository-level refactoring, and automated debugging.

## Links

- Models: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- Agent framework: [github.com/zenlm/zen-agent](https://github.com/zenlm/zen-agent)
- Docs: [zenlm.org](https://zenlm.org)

## License

Apache 2.0 — Copyright 2024 Zen LM Authors