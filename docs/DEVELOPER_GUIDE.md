# nanochat Developer Guide

This comprehensive guide is for developers who want to understand, modify, or extend nanochat. It covers the codebase architecture, design patterns, and step-by-step instructions for common development tasks.

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Codebase Overview](#codebase-overview)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Key Components Deep Dive](#key-components-deep-dive)
5. [How Training Works](#how-training-works)
6. [How Inference Works](#how-inference-works)
7. [Adding New Features](#adding-new-features)
8. [Testing](#testing)
9. [Debugging Tips](#debugging-tips)
10. [Contributing Guidelines](#contributing-guidelines)

---

## Development Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

### Step 2: Install Development Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with GPU support
uv sync --extra gpu

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Install Development Tools

The project uses pytest for testing. It's included in the dev dependencies:

```bash
# Verify pytest is available
pytest --version
```

### Step 4: Set Up Your Editor

**For VS Code:**
1. Install the Python extension
2. Select the `.venv/bin/python` interpreter
3. Configure `python.linting.enabled: true`

**For PyCharm:**
1. Go to Settings → Project → Python Interpreter
2. Add the `.venv` interpreter

### Step 5: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Try a quick training test
python -m scripts.base_train --depth=4 --num-iterations=2 --core-metric-every=-1 --sample-every=-1
```

---

## Codebase Overview

### Directory Structure

```
nanochat/
├── nanochat/              # Core library modules
│   ├── __init__.py        # Empty package marker
│   ├── gpt.py             # GPT model architecture (THE MAIN MODEL)
│   ├── engine.py          # Inference engine with KV cache
│   ├── tokenizer.py       # BPE tokenizer (rustbpe + tiktoken)
│   ├── dataloader.py      # Distributed data loading
│   ├── dataset.py         # Data downloading and preprocessing
│   ├── optim.py           # MuonAdamW optimizer
│   ├── checkpoint_manager.py  # Save/load checkpoints
│   ├── common.py          # Utilities (logging, distributed)
│   ├── core_eval.py       # CORE benchmark evaluation
│   ├── loss_eval.py       # Bits-per-byte evaluation
│   ├── execution.py       # Python tool execution
│   ├── report.py          # Training report generation
│   ├── flash_attention.py # Flash Attention wrapper
│   ├── ui.html            # Web UI (HTML/CSS/JS)
│   └── logo.svg           # nanochat logo
│
├── scripts/               # Executable scripts
│   ├── base_train.py      # Pretraining script
│   ├── base_eval.py       # Base model evaluation
│   ├── chat_sft.py        # Supervised fine-tuning
│   ├── chat_eval.py       # Chat model evaluation
│   ├── chat_rl.py         # Reinforcement learning
│   ├── chat_cli.py        # CLI chat interface
│   ├── chat_web.py        # Web chat server
│   ├── tok_train.py       # Tokenizer training
│   └── tok_eval.py        # Tokenizer evaluation
│
├── tasks/                 # Evaluation task implementations
│   ├── common.py          # TaskMixture, TaskSequence base classes
│   ├── arc.py             # ARC benchmark
│   ├── gsm8k.py           # GSM8K math benchmark
│   ├── mmlu.py            # MMLU benchmark
│   ├── humaneval.py       # HumanEval coding benchmark
│   ├── smoltalk.py        # SmolTalk conversation dataset
│   ├── spellingbee.py     # SpellingBee task
│   └── customjson.py      # Custom JSONL dataset loader
│
├── runs/                  # Shell scripts for workflows
│   ├── speedrun.sh        # Full training pipeline
│   ├── miniseries.sh      # Scaling law experiments
│   ├── scaling_laws.sh    # Research experiments
│   └── runcpu.sh          # CPU/MPS testing
│
├── tests/                 # Test files
│   ├── test_engine.py     # Engine tests
│   └── test_attention_fallback.py
│
├── dev/                   # Development utilities
│   ├── gen_synthetic_data.py
│   ├── repackage_data_reference.py
│   └── *.ipynb            # Analysis notebooks
│
├── pyproject.toml         # Project configuration
├── uv.lock                # Dependency lock file
└── README.md              # Main documentation
```

### File Purpose Summary

| File | Purpose | When to Modify |
|------|---------|----------------|
| `gpt.py` | Model architecture | Changing model design |
| `engine.py` | Inference with KV cache | Adding generation features |
| `tokenizer.py` | Text tokenization | Changing vocabulary |
| `dataloader.py` | Data loading | Changing data pipeline |
| `optim.py` | Optimizer | Changing training dynamics |
| `base_train.py` | Pretraining loop | Modifying training |
| `chat_sft.py` | Fine-tuning loop | Adding SFT features |
| `chat_web.py` | Web server | Adding API endpoints |

---

## Understanding the Architecture

### The GPT Model (`nanochat/gpt.py`)

The model follows the standard transformer decoder architecture with modern improvements:

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT Model                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Token IDs [batch, seq_len]                          │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Token Embedding (wte)                                │   │
│  │  vocab_size → n_embd                                 │   │
│  │  + Norm (no learnable params)                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Transformer Blocks × n_layer                         │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Causal Self-Attention                          │  │   │
│  │  │  - Query/Key/Value projections                  │  │   │
│  │  │  - Rotary Position Embeddings (RoPE)           │  │   │
│  │  │  - QK Normalization                             │  │   │
│  │  │  - Flash Attention 3 (or SDPA fallback)        │  │   │
│  │  │  - Optional: Sliding Window Attention          │  │   │
│  │  │  - Optional: Value Embeddings (ResFormer)      │  │   │
│  │  │  - Group Query Attention (GQA)                  │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  MLP (Feed-Forward)                             │  │   │
│  │  │  - Linear: n_embd → 4*n_embd                   │  │   │
│  │  │  - ReLU² activation                             │  │   │
│  │  │  - Linear: 4*n_embd → n_embd                   │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  + Residual Connections (with resid_lambdas)        │   │
│  │  + x0 Skip Connection (with x0_lambdas)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Final Norm + LM Head                                │   │
│  │  n_embd → vocab_size                                │   │
│  │  + Logit soft-capping (15)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│  Output: Logits [batch, seq_len, vocab_size]               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Model Features

1. **Rotary Position Embeddings (RoPE)**
   - No learned positional embeddings
   - Relative position encoding via rotation
   - Better length generalization

2. **QK Normalization**
   - Normalizes queries and keys before attention
   - Improves training stability

3. **ReLU² Activation**
   - `ReLU(x)²` instead of GELU
   - Simpler and surprisingly effective

4. **Sliding Window Attention**
   - Configurable via `--window-pattern`
   - "SSSL" = Short, Short, Short, Long pattern
   - Reduces memory for long sequences

5. **Value Embeddings (ResFormer)**
   - Adds token embeddings directly to values
   - On alternating layers
   - Improves gradient flow

6. **Flash Attention 3**
   - Uses FA3 on Hopper+ GPUs
   - Falls back to PyTorch SDPA elsewhere

### The Optimizer (`nanochat/optim.py`)

nanochat uses a hybrid optimizer combining Muon and AdamW:

```
┌─────────────────────────────────────────────────────────────┐
│                    MuonAdamW Optimizer                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Parameter Groups:                                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  MUON (for matrix parameters)                       │     │
│  │  - Transformer weights (c_q, c_k, c_v, c_proj, etc) │     │
│  │  - Uses Newton-Schulz iterations                    │     │
│  │  - Momentum-based updates                           │     │
│  │  - Weight decay scheduled                           │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ADAMW (for embedding parameters)                   │     │
│  │  - wte (token embeddings)                          │     │
│  │  - lm_head (output layer)                          │     │
│  │  - value_embeds (ResFormer)                        │     │
│  │  - resid_lambdas, x0_lambdas (scalars)            │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Learning Rate Scaling:                                      │
│  - AdamW params: LR scales by 1/√(dmodel/768)               │
│  - All params: LR scales by √(batch_size/2^19)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Download (dataset.py)                                    │
│     ┌──────────┐                                            │
│     │ FineWeb- │ → Download shards from HuggingFace          │
│     │ Edu      │                                            │
│     └──────────┘                                            │
│                                                              │
│  2. Tokenize (dataloader.py)                                │
│     ┌──────────┐                                            │
│     │ Raw Text │ → BPE Tokenizer → Token IDs                │
│     └──────────┘                                            │
│                                                              │
│  3. Pack (BOS-aligned bestfit)                              │
│     - Each batch row starts with BOS                        │
│     - Documents packed to minimize padding                   │
│     - Best-fit algorithm selects documents                   │
│                                                              │
│  4. Distribute                                               │
│     - Each GPU rank gets different data                      │
│     - Synchronized via DDP                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components Deep Dive

### GPTConfig Dataclass

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # Maximum sequence length
    vocab_size: int = 32768       # Vocabulary size
    n_layer: int = 12             # Number of transformer layers
    n_head: int = 6               # Number of attention heads
    n_kv_head: int = 6            # Number of KV heads (for GQA)
    n_embd: int = 768             # Model dimension
    window_pattern: str = "SSSL"  # Sliding window pattern
```

### The Attention Mechanism

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 1. Project inputs to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. Add value embeddings (ResFormer)
        if ve is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :channels]))
            v = v + gate.unsqueeze(-1) * ve

        # 3. Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 4. QK normalization
        q, k = norm(q), norm(k)

        # 5. Flash Attention
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        # 6. Output projection
        return self.c_proj(y)
```

### The Inference Engine

```python
class Engine:
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        # 1. Prefill: Process prompt with batch=1
        kv_cache_prefill = KVCache(batch_size=1, ...)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)

        # 2. Clone KV cache for each sample
        kv_cache_decode = KVCache(batch_size=num_samples, ...)
        kv_cache_decode.prefill(kv_cache_prefill)

        # 3. Generate loop
        for _ in range(max_tokens):
            # Sample next token
            next_ids = sample_next_token(logits, rng, temperature, top_k)

            # Handle tool calls (calculator)
            if token == python_start:
                state.in_python_block = True
            elif token == python_end:
                result = use_calculator(expr)
                state.forced_tokens.extend(result_tokens)

            yield token_column, token_masks

            # Get next logits
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)
```

### The KV Cache

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, ...)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, ...)
        # Track position per batch element
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, ...)

    def advance(self, num_tokens):
        """Move position forward after processing tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """Copy another cache's contents (for multi-sample generation)."""
        self.k_cache[:, :, :pos] = other.k_cache[:, :, :pos]
        self.v_cache[:, :, :pos] = other.v_cache[:, :, :pos]
        self.cache_seqlens.fill_(other.get_pos())
```

---

## How Training Works

### Pretraining Flow (`scripts/base_train.py`)

```python
# 1. Initialize model
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# 2. Set up optimizer
optimizer = model.setup_optimizer(...)

# 3. Set up data loader
train_loader = tokenizing_distributed_data_loader_bos_bestfit(...)

# 4. Training loop
for step in range(num_iterations):
    # Evaluation (periodic)
    if step % eval_every == 0:
        val_bpb = evaluate_bpb(model, val_loader, ...)

    # Forward + backward
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        (loss / grad_accum_steps).backward()
        x, y, _ = next(train_loader)

    # Update weights
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Save checkpoint (periodic)
    if step % save_every == 0:
        save_checkpoint(...)
```

### SFT Flow (`scripts/chat_sft.py`)

```python
# 1. Load pretrained model
model, tokenizer, meta = load_model("base", device, phase="train")

# 2. Prepare SFT dataset
train_dataset = TaskMixture([
    SmolTalk(split="train"),
    MMLU(subset="auxiliary_train", split="train"),
    GSM8K(subset="main", split="train"),
    CustomJSON(filepath=identity_path),
    SpellingBee(size=80000, split="train"),
])

# 3. SFT data generator
def sft_data_generator_bos_bestfit(split):
    while True:
        # Pack conversations into batches
        for _ in range(batch_size):
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            # Best-fit packing into rows
            ...
        yield inputs, targets

# 4. Fine-tuning loop (similar to pretraining)
```

### Key Training Concepts

**Gradient Accumulation:**
When `total_batch_size` > `device_batch_size * world_size`, we accumulate gradients:
```python
grad_accum_steps = total_batch_size // (device_batch_size * world_size)
for micro_step in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps
    loss.backward()  # Gradients accumulate
# Then optimizer.step() uses the accumulated gradients
```

**Learning Rate Schedule:**
```python
def get_lr_multiplier(it):
    warmup_iters = warmup_ratio * num_iterations
    warmdown_iters = warmdown_ratio * num_iterations

    if it < warmup_iters:
        return (it + 1) / warmup_iters  # Linear warmup
    elif it <= num_iterations - warmdown_iters:
        return 1.0  # Constant
    else:
        # Linear cooldown to final_lr_frac
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * final_lr_frac
```

---

## How Inference Works

### CLI Chat (`scripts/chat_cli.py`)

```python
# Load model
model, tokenizer, _ = load_model("sft", device, phase="eval")
engine = Engine(model, tokenizer)

# Conversation loop
conversation_tokens = [bos]
while True:
    user_input = input("User: ")

    # Add user message
    conversation_tokens.extend([user_start] + tokenizer.encode(user_input) + [user_end])
    conversation_tokens.append(assistant_start)

    # Generate response
    for token_column, _ in engine.generate(conversation_tokens, ...):
        print(tokenizer.decode([token_column[0]]), end="")

    # Add response to history
    conversation_tokens.extend(response_tokens)
```

### Web Server (`scripts/chat_web.py`)

```python
# FastAPI app with worker pool
class WorkerPool:
    def __init__(self, num_gpus):
        # Load model on each GPU
        for gpu_id in range(num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            model, tokenizer, _ = load_model(source, device, ...)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id, device, engine, tokenizer, ...)
            self.workers.append(worker)
            self.available_workers.put(worker)

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    # Acquire worker from pool
    worker = await worker_pool.acquire_worker()

    # Build conversation tokens
    conversation_tokens = build_tokens(request.messages)

    # Stream response
    async def stream_and_release():
        async for chunk in generate_stream(worker, conversation_tokens, ...):
            yield chunk
        await worker_pool.release_worker(worker)

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")
```

---

## Adding New Features

### Adding a New Evaluation Task

1. **Create task file in `tasks/`:**

```python
# tasks/my_task.py
from tasks.common import Task
from datasets import load_dataset

class MyTask(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("my_dataset", split=split)

    @property
    def eval_type(self):
        return 'generative'  # or 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        return {
            "messages": [
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": row['answer']},
            ]
        }

    def evaluate(self, conversation, assistant_response):
        # Return 1 if correct, 0 if wrong
        expected = conversation['messages'][-1]['content']
        return int(assistant_response.strip() == expected.strip())
```

2. **Add to `scripts/chat_eval.py`:**

```python
from tasks.my_task import MyTask

task_module = {
    'MyTask': MyTask,
    # ... existing tasks
}
```

### Adding a New Model Architecture Feature

1. **Modify `nanochat/gpt.py`:**

```python
class GPTConfig:
    # Add new config option
    my_new_feature: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Add new layers if needed
        if config.my_new_feature:
            self.my_layer = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, ...):
        # Use new feature
        if hasattr(self, 'my_layer'):
            x = self.my_layer(x)
```

2. **Update checkpoint loading in `checkpoint_manager.py`:**

```python
def _patch_missing_keys(model_data, model_config):
    # Handle loading old checkpoints without new feature
    if "my_layer.weight" not in model_data and model_config.my_new_feature:
        model_data["my_layer.weight"] = torch.zeros(...)
```

### Adding a New Training Script

```python
# scripts/my_train.py
"""
My custom training script.
Run as: python -m scripts.my_train
"""
import argparse
from nanochat.common import compute_init, compute_cleanup
from nanochat.checkpoint_manager import load_model, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--my-arg', type=int, default=10)
args = parser.parse_args()

# Initialize
ddp, rank, local_rank, world_size, device = compute_init("cuda")

# Load model
model, tokenizer, meta = load_model("sft", device, phase="train")

# Your training loop
for step in range(args.my_arg):
    # ...
    pass

# Save
save_checkpoint(...)

# Cleanup
compute_cleanup()
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_engine.py -v

# Specific test function
pytest tests/test_engine.py::test_seed_reproducibility -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### Writing Tests

```python
# tests/test_my_feature.py
import torch
from nanochat.gpt import GPT, GPTConfig

def test_my_feature():
    """Test that my feature works correctly."""
    config = GPTConfig(n_layer=2, n_embd=64)
    model = GPT(config)
    model.init_weights()

    # Test with dummy input
    x = torch.randint(0, 100, (2, 16))
    output = model(x)

    assert output.shape == (2, 16, config.vocab_size)

@pytest.mark.slow
def test_slow_feature():
    """This test takes a while to run."""
    # ...
```

---

## Debugging Tips

### Memory Issues

```python
# Check GPU memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()

# Find memory hogs
for name, param in model.named_parameters():
    print(f"{name}: {param.numel() * param.element_size() / 1e6:.2f} MB")
```

### Gradient Issues

```python
# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")

# Gradient clipping (if needed)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Distributed Debugging

```python
# Only print on rank 0
from nanochat.common import print0
print0("This only prints on rank 0")

# Check if distributed is working
ddp, rank, local_rank, world_size = get_dist_info()
print(f"Rank {rank}/{world_size} on device {local_rank}")
```

### Tokenizer Debugging

```python
# Visualize tokenization
tokenizer = get_tokenizer()
text = "Hello world!"
ids = tokenizer.encode(text)
print(f"IDs: {ids}")
print(f"Decoded: {tokenizer.decode(ids)}")

# Visualize with conversation
ids, mask = tokenizer.render_conversation(conversation)
print(tokenizer.visualize_tokenization(ids, mask))
```

---

## Contributing Guidelines

### Before Submitting a PR

1. **Run tests:** `pytest tests/ -v`
2. **Check formatting:** Follow existing code style
3. **Document changes:** Update docstrings and comments
4. **Test on GPU if possible:** Many features are GPU-specific

### Commit Messages

Follow this format:
```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain the problem this commit solves.

- Bullet points are okay
- Keep them concise
```

### AI Disclosure Policy

Per the project's guidelines, declare any parts that had substantial LLM contribution that you don't fully understand.

### Pull Request Template

```markdown
## Summary
Brief description of changes.

## Changes
- List specific changes
- One per line

## Testing
How was this tested?

## AI Disclosure
Did any AI tools help write this code? If so, which parts?
```

---

## Further Reading

- [Architecture Guide](ARCHITECTURE.md) - Deep technical details
- [Quick Start](QUICK_START.md) - Getting started quickly
- [User Guide](USER_GUIDE.md) - Using nanochat
- [README](../README.md) - Project overview

---

*This guide is part of the nanochat documentation.*
