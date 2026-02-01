# nanochat User Guide

This comprehensive guide covers everything you need to know to use nanochat effectively.

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation Step by Step](#installation-step-by-step)
4. [Understanding the Pipeline](#understanding-the-pipeline)
5. [Chatting with Your Model](#chatting-with-your-model)
6. [Training Your Own Model](#training-your-own-model)
7. [Evaluating Model Performance](#evaluating-model-performance)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)
10. [Glossary](#glossary)

---

## Introduction

nanochat is a complete, educational system for training and using ChatGPT-like AI models. Created by Andrej Karpathy, it demonstrates the entire pipeline from raw text to a conversational AI.

---

## System Requirements

### Minimum (Using Pre-trained Models)
- Python 3.10+
- 8 GB RAM
- 10 GB disk space

### Recommended (Training Small Models)
- 32 GB RAM
- 100 GB SSD
- NVIDIA RTX 3080 or better

### Optimal (Full GPT-2 Training)
- 256 GB RAM
- 500 GB NVMe SSD
- 8× NVIDIA H100 80GB

---

## Installation Step by Step

### Step 1: Download nanochat
```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

### Step 2: Install uv Package Manager
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

### Step 3: Create Virtual Environment
```bash
# For NVIDIA GPU:
uv sync --extra gpu

# For CPU only:
uv sync --extra cpu
```

### Step 4: Activate Virtual Environment
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Your prompt should now show `(.venv)` at the beginning.

---

## Understanding the Pipeline

nanochat follows five stages:

1. **Tokenizer Training**: Convert text to numbers
2. **Pretraining**: Learn language patterns from web text
3. **SFT (Supervised Fine-Tuning)**: Learn to follow instructions
4. **Evaluation**: Measure performance on benchmarks
5. **Inference**: Generate responses to user prompts

---

## Chatting with Your Model

### Command Line Interface (CLI)

**Interactive mode:**
```bash
python -m scripts.chat_cli
```

**Single question:**
```bash
python -m scripts.chat_cli -p "What is the speed of light?"
```

**With custom settings:**
```bash
python -m scripts.chat_cli \
    -p "Write a poem" \
    -t 1.0 \              # Temperature (creativity)
    -k 50                  # Top-k sampling
```

### Web Interface

**Start the server:**
```bash
python -m scripts.chat_web
```

**Open in browser:** http://localhost:8000

**Server options:**
```bash
python -m scripts.chat_web \
    --port 3000 \           # Different port
    --temperature 0.8 \     # Default temperature
    --max-tokens 512 \      # Max response length
    --num-gpus 4            # Use multiple GPUs
```

### Understanding Parameters

| Parameter | What It Does | Range | Default |
|-----------|--------------|-------|---------|
| `temperature` | Controls randomness. Low = predictable, High = creative | 0.0 - 2.0 | 0.6 (CLI), 0.8 (Web) |
| `top_k` | Limits vocabulary to top K tokens | 0 - 200 | 50 |
| `max_tokens` | Maximum response length | 1 - 4096 | 256 (CLI), 512 (Web) |

---

## Training Your Own Model

### Quick Training (CPU, ~10 minutes)

```bash
# Train tokenizer first
python -m scripts.tok_train

# Train a tiny model
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=512 \
    --num-iterations=20 \
    --core-metric-every=-1
```

### Single GPU Training (~4 hours)

```bash
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8
```

### Full 8×GPU Training (~3 hours)

```bash
# All-in-one script
bash runs/speedrun.sh

# Or manually:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=12

# Then fine-tune:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

### Training Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--depth` | Model layers (4-30+). More = smarter but slower | 20 |
| `--device-batch-size` | Examples per GPU per step. Lower if OOM | 32 |
| `--total-batch-size` | Total examples per optimization step | 524288 tokens |
| `--num-iterations` | Number of training steps | Auto-calculated |
| `--target-param-data-ratio` | Data tokens per parameter | 10.5 |
| `--run` | Experiment name for W&B. "dummy" = no logging | "dummy" |

### Monitoring Training

Training output shows:
```
step 00100/16704 (0.60%) | loss: 8.12 | lrm: 1.00 | dt: 650ms | tok/sec: 806K | mfu: 45.2 | epoch: 1 | total time: 1.08m | eta: 178.5m
```

- **loss**: Should decrease (model is learning)
- **lrm**: Learning rate multiplier
- **dt**: Time per step (milliseconds)
- **tok/sec**: Tokens processed per second
- **mfu**: Model FLOPs Utilization (40-50% is good)
- **eta**: Estimated time remaining

### Where Are Checkpoints Saved?

```
~/.cache/nanochat/
├── base_checkpoints/       # Pretrained models
│   └── d24/
│       ├── model_016704.pt
│       ├── meta_016704.json
│       └── optim_016704_rank0.pt
├── sft_checkpoints/        # Fine-tuned models
├── tokenizer/              # Trained tokenizer
└── data/                   # Downloaded training data
```

---

## Evaluating Model Performance

### Base Model Evaluation

```bash
# Full evaluation (CORE, BPB, samples)
python -m scripts.base_eval

# Specific evaluations
python -m scripts.base_eval --eval core       # Just CORE metric
python -m scripts.base_eval --eval bpb        # Just bits-per-byte
python -m scripts.base_eval --eval sample     # Just samples
```

### Chat Model Evaluation

```bash
# All benchmarks
python -m scripts.chat_eval -i sft

# Specific benchmark
python -m scripts.chat_eval -i sft -a ARC-Easy
python -m scripts.chat_eval -i sft -a MMLU
python -m scripts.chat_eval -i sft -a GSM8K
```

### Understanding Metrics

| Metric | What It Measures | GPT-2 Target |
|--------|-----------------|--------------|
| CORE | Overall capability (0-1 scale) | 0.256 |
| BPB | Compression efficiency (lower = better) | ~1.0 |
| ARC-Easy | Science questions accuracy | ~50% |
| ARC-Challenge | Harder science questions | ~25% |
| MMLU | General knowledge accuracy | ~30% |
| GSM8K | Math problem accuracy | ~10% |

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NANOCHAT_BASE_DIR` | Where to store data/checkpoints | `~/.cache/nanochat` |
| `OMP_NUM_THREADS` | OpenMP thread count | System default |
| `WANDB_RUN` | W&B experiment name (for speedrun.sh) | `dummy` |

### Common CLI Arguments

**All training scripts:**
- `--run NAME`: W&B run name ("dummy" = no logging)
- `--device-type TYPE`: cuda, cpu, or mps
- `--device-batch-size N`: Batch size per GPU

**Base training (`base_train.py`):**
- `--depth N`: Model depth (layers)
- `--num-iterations N`: Training steps
- `--eval-every N`: Evaluate every N steps
- `--save-every N`: Checkpoint every N steps

**Chat scripts:**
- `-t, --temperature FLOAT`: Sampling temperature
- `-k, --top-k INT`: Top-k sampling
- `-p, --prompt TEXT`: Input prompt (CLI)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'nanochat'"

**Cause:** Virtual environment not activated or wrong directory.

**Solution:**
```bash
cd /path/to/nanochat
source .venv/bin/activate
```

### "CUDA out of memory"

**Cause:** Batch size too large for GPU memory.

**Solution:**
```bash
python -m scripts.base_train --device-batch-size=8  # Try 8, 4, 2, or 1
```

### "Tokenizer not found"

**Cause:** Tokenizer hasn't been trained yet.

**Solution:**
```bash
python -m nanochat.dataset -n 2  # Download data first
python -m scripts.tok_train      # Train tokenizer
```

### "No checkpoints found"

**Cause:** Model hasn't been trained yet.

**Solution:**
```bash
# Train a quick model first:
python -m scripts.base_train --depth=4 --num-iterations=20
```

### Web interface won't load

**Possible causes:**
1. Port already in use: Try `--port 3001`
2. Firewall blocking: Allow port 8000
3. Model not loaded: Check terminal for errors

### Training loss not decreasing

**Possible causes:**
1. Learning rate too high: Training is unstable
2. Model too small: Not enough capacity
3. Data issue: Check data loading logs

---

## Glossary

| Term | Definition |
|------|------------|
| **BPE** | Byte Pair Encoding - algorithm for creating tokenizer vocabulary |
| **Checkpoint** | Saved model state that can be loaded later |
| **CORE** | Benchmark suite measuring model capabilities |
| **DDP** | Distributed Data Parallel - training across multiple GPUs |
| **Epoch** | One complete pass through training data |
| **GQA** | Group Query Attention - efficient attention mechanism |
| **Loss** | Measure of model error (lower is better) |
| **LR** | Learning Rate - step size for optimization |
| **MFU** | Model FLOPs Utilization - GPU efficiency percentage |
| **SFT** | Supervised Fine-Tuning - teaching model to follow instructions |
| **Token** | Unit of text (word, subword, or character) |
| **Transformer** | Neural network architecture used in LLMs |
| **W&B** | Weights & Biases - experiment tracking platform |

---

## Getting Help

- **DeepWiki**: https://deepwiki.com/karpathy/nanochat
- **GitHub Discussions**: https://github.com/karpathy/nanochat/discussions
- **Discord**: #nanochat channel

---

*This guide is part of the nanochat documentation. See also: [Quick Start](QUICK_START.md), [Developer Guide](DEVELOPER_GUIDE.md), [Architecture](ARCHITECTURE.md)*
