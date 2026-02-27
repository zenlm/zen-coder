---
license: apache-2.0
language:
- en
tags:
- zen
- zen-lm
- code
- coding
- agentic
- moe
library_name: transformers
pipeline_tag: text-generation
---

<p align="center">
  <img src="https://zenlm.org/logo.png" width="300"/>
</p>

<h1 align="center">Zen Coder</h1>

<p align="center">
  <strong>Agentic coding AI by Zen LM — from edge to frontier</strong>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/zenlm/zen-coder-480b-instruct">HuggingFace</a> &nbsp;|&nbsp;
  📖 <a href="https://zenlm.org">Docs</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/zenlm">GitHub</a>
</p>

---

## Introduction

**Zen Coder** is Zen LM's family of code-focused AI models, spanning three capability tiers from edge deployment to frontier performance. The flagship model, `zen-coder-480b-instruct`, is a 480B-parameter Mixture of Experts (MoE) model with 35B active parameters, delivering state-of-the-art results on agentic coding, browser-use, and tool-use benchmarks.

## Model Family

| Model | Parameters | Active | Context | Use Case |
|-------|------------|--------|---------|----------|
| [zen-coder-480b-instruct](https://huggingface.co/zenlm/zen-coder-480b-instruct) | 480B MoE | 35B | 256K | Frontier agentic coding |
| [zen-coder-flash](https://huggingface.co/zenlm/zen-coder-flash) | 31B MoE | 3B | 131K | Balanced performance |
| [zen-coder](https://huggingface.co/zenlm/zen-coder) | 4B | 4B | 32K | Edge / mobile |

## Highlights

- **Agentic coding**: state-of-the-art on SWE-bench, comparable to frontier closed models
- **256K context**: native support, extendable to 1M tokens via Yarn — handles full repository scale
- **358 programming languages**: full spectrum from ABAP to Zig
- **Fill-in-the-middle (FIM)**: code insertion at the cursor position
- **Function calling**: structured tool-use with a dedicated tool parser
- **Long-context reasoning**: extended chain-of-thought for complex engineering tasks

## Quick Start

### Install

```bash
pip install transformers torch
```

### Chat with Zen Coder (480B Flagship)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "zenlm/zen-coder-480b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Write a quicksort algorithm in Python with type hints."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Fill-in-the-Middle (FIM)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "zenlm/zen-coder-480b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()

# Structure: prefix + suffix + middle token
input_text = """<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>"""

messages = [
    {"role": "system", "content": "You are a code completion assistant."},
    {"role": "user", "content": input_text}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

eos_token_ids = [151659, 151661, 151662, 151663, 151664, 151643, 151645]
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=False,
    eos_token_id=eos_token_ids
)[0]

output_text = tokenizer.decode(
    generated_ids[len(model_inputs.input_ids[0]):],
    skip_special_tokens=True
)
print(output_text)
```

### Edge Deployment (4B Model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "zenlm/zen-coder"  # 4B, 32K context

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [{"role": "user", "content": "Write a binary search in Rust."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True))
```

## Supported Languages (358 total)

```
ABAP, ActionScript, Ada, Agda, Alloy, ApacheConf, AppleScript, Arduino,
Assembly, Awk, Batchfile, C, C#, C++, COBOL, CSS, Clojure, CoffeeScript,
Dart, Dockerfile, Elixir, Elm, Erlang, F#, FORTRAN, Go, GraphQL, Groovy,
HTML, Haskell, Java, JavaScript, Julia, Kotlin, Lua, Makefile, Markdown,
NSIS, Nginx, OCaml, Objective-C, PHP, Pascal, Perl, PowerShell, Prolog,
Python, R, Racket, Ruby, Rust, SQL, Scala, Shell, Solidity, Swift,
TypeScript, VHDL, Verilog, Vue, WebAssembly, YAML, Zig, and 300+ more
```

## Performance Benchmarks

| Benchmark | zen-coder-480b | zen-coder-flash |
|-----------|---------------|-----------------|
| SWE-bench Verified | State-of-the-art | 59.2% |
| AIME 2025 | Competitive | 91.6% |
| GPQA | Competitive | 75.2% |
| t²-Bench | Competitive | 79.5% |

## Hardware Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| zen-coder (4B) | 8GB | Full precision, single GPU |
| zen-coder-flash (31B MoE) | 24GB | Efficient MoE, 3B active |
| zen-coder-480b-instruct | 8x 80GB | Tensor parallel recommended |

## License

Apache 2.0

## Citation

```bibtex
@misc{zenlm2025zen-coder,
    title={Zen Coder: Agentic Coding AI by Zen LM},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-coder-480b-instruct}}
}
```

---

<p align="center">
  <strong>Zen LM by Hanzo AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/zenlm">HuggingFace</a> &nbsp;|&nbsp;
  <a href="https://github.com/zenlm">GitHub</a>
</p>
