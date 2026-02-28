.PHONY: help train train-mlx train-full fuse test serve clean

MODEL_NAME = zen-coder
BASE_MODEL_HF = Qwen/Qwen3-Coder-Next
BASE_MODEL_MLX = lmstudio-community/Qwen3-Coder-Next-MLX-4bit
HF_REPO = zenlm/$(MODEL_NAME)
OUTPUT_DIR = ./training/output
ADAPTER_DIR = ./training/output/mlx-adapters
FUSED_DIR = ./training/output/zen-coder-mlx

help:
	@echo "Zen Coder - Flagship Coding Model"
	@echo "=================================="
	@echo "  make train       LoRA fine-tune on MLX (Apple Silicon)"
	@echo "  make train-full  Full training on agentic dataset"
	@echo "  make fuse        Fuse LoRA adapters into model"
	@echo "  make test        Test model identity + coding"
	@echo "  make serve       Start local MLX inference server"
	@echo "  make clean       Clean training artifacts"

train:
	@echo "Training zen-coder on MLX..."
	python training/train_zen_coder_4b_mlx.py --epochs 3 --batch-size 4

train-mlx:
	@echo "MLX LoRA training on Qwen3-Coder-Next..."
	python -m mlx_lm.lora \
		--model $(BASE_MODEL_MLX) \
		--train \
		--data training/data \
		--adapter-path $(ADAPTER_DIR) \
		--iters 500 \
		--batch-size 2 \
		--num-layers 16 \
		--learning-rate 1e-5

train-full:
	python training/train_full.py

fuse:
	python -m mlx_lm.fuse \
		--model $(BASE_MODEL_MLX) \
		--adapter-path $(ADAPTER_DIR) \
		--save-path $(FUSED_DIR)

test:
	@echo "Testing zen-coder..."
	python -c "from mlx_lm import load, generate; \
		model, tok = load('$(BASE_MODEL_MLX)'); \
		print(generate(model, tok, prompt='Who are you?', max_tokens=100))"

serve:
	@echo "Starting zen-coder server on port 3690..."
	python -m mlx_lm.server --model $(BASE_MODEL_MLX) --port 3690

clean:
	rm -rf $(OUTPUT_DIR)

.DEFAULT_GOAL := help
