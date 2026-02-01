# nanochat Architecture Guide

This document provides deep technical details about nanochat's architecture. It's intended for developers who want to understand the internals of the system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Tokenization System](#tokenization-system)
4. [Data Pipeline](#data-pipeline)
5. [Optimization](#optimization)
6. [Distributed Training](#distributed-training)
7. [Inference System](#inference-system)
8. [Evaluation System](#evaluation-system)
9. [Web Server Architecture](#web-server-architecture)
10. [Design Decisions](#design-decisions)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           nanochat Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Dataset   │ -> │  Tokenizer  │ -> │   Model     │ -> │  Inference  │  │
│  │ (FineWeb)   │    │ (BPE/tiktoken)   │   (GPT)     │    │  (Engine)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ DataLoader  │    │    Vocab    │    │ Checkpoint  │    │    CLI/     │  │
│  │(distributed)│    │  (32768)    │    │  Manager    │    │   Web UI    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  Training Infrastructure:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MuonAdamW Optimizer + DDP + Learning Rate Scheduler + W&B Logging  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Dependencies

```
                         ┌─────────────┐
                         │   common    │
                         │  (utils)    │
                         └──────┬──────┘
                                │
        ┌───────────────┬───────┴───────┬───────────────┐
        │               │               │               │
        ▼               ▼               ▼               ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ tokenizer │  │  dataset  │  │    gpt    │  │checkpoint │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  │  manager  │
        │              │              │        └─────┬─────┘
        └──────────────┴──────┬───────┴──────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  dataloader   │
                      └───────┬───────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
  ┌───────────┐                              ┌───────────┐
  │base_train │                              │   engine  │
  │ chat_sft  │                              │           │
  │ chat_rl   │                              └─────┬─────┘
  └───────────┘                                    │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │  chat_cli / chat_web     │
                                    └──────────────────────────┘
```

---

## Model Architecture

### GPT Configuration

The model is configured via `GPTConfig`:

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048    # Maximum context window
    vocab_size: int = 32768     # Vocabulary size
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 6             # Query heads per layer
    n_kv_head: int = 6          # Key/Value heads (for GQA)
    n_embd: int = 768           # Hidden dimension
    window_pattern: str = "SSSL" # Sliding window pattern
```

### Model Size Calculation

Model dimensions are derived from depth:
```
model_dim = depth × aspect_ratio (default 64)
model_dim = round_up(model_dim, head_dim)  # Ensure divisibility
num_heads = model_dim / head_dim (default 128)
```

Example configurations:
| Depth | model_dim | num_heads | Parameters |
|-------|-----------|-----------|------------|
| 4     | 256       | 2         | ~3M        |
| 12    | 768       | 6         | ~125M      |
| 20    | 1280      | 10        | ~350M      |
| 24    | 1536      | 12        | ~1.6B      |

### Layer Structure

Each transformer block contains:

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Pre-norm architecture with residual connections
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

### Attention Implementation

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Projections (no bias)
        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Value embedding gate (ResFormer, alternating layers)
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(32, n_kv_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project to Q, K, V with layout (B, T, H, D) for FA3
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): input-dependent gated addition
        if ve is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))  # Range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK normalization
        q, k = norm(q), norm(k)

        # Flash Attention
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)

        return self.c_proj(y.view(B, T, -1))
```

### Rotary Position Embeddings

```python
def apply_rotary_emb(x, cos, sin):
    """Apply RoPE by rotating pairs of dimensions."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def _precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    """Precompute cos/sin for all positions."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().bfloat16(), freqs.sin().bfloat16()
```

### MLP Layer

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² activation
        x = self.c_proj(x)
        return x
```

### Sliding Window Attention

```python
def _compute_window_sizes(config):
    """Compute per-layer window sizes from pattern string."""
    pattern = config.window_pattern.upper()  # e.g., "SSSL"
    long_window = config.sequence_len
    short_window = long_window // 2

    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        if char == 'L':
            window_sizes.append((long_window, 0))  # Full context
        else:  # 'S'
            window_sizes.append((short_window, 0))  # Half context

    # Final layer always gets full context
    window_sizes[-1] = (long_window, 0)
    return window_sizes
```

### Forward Pass

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Rotary embeddings with cache offset
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # Token embedding + norm
    x = self.transformer.wte(idx)
    x = norm(x)
    x0 = x  # Save for x0 residual

    # Transformer blocks
    for i, block in enumerate(self.transformer.h):
        # Per-layer scaling
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

    # Final norm + LM head
    x = norm(x)
    logits = self.lm_head(x)[..., :self.config.vocab_size]

    # Logit soft-capping
    logits = 15 * torch.tanh(logits.float() / 15)

    if targets is not None:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ...)
    return logits
```

---

## Tokenization System

### Tokenizer Architecture

nanochat uses a two-component tokenizer:
1. **rustbpe**: Fast BPE training in Rust
2. **tiktoken**: Efficient inference from OpenAI

```python
class RustBPETokenizer:
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # Train with rustbpe
        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(text_iterator, vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)

        # Create tiktoken encoding for inference
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        special_tokens = {name: len(mergeable_ranks) + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
        return cls(enc, "<|bos|>")
```

### Special Tokens

```python
SPECIAL_TOKENS = [
    "<|bos|>",           # Beginning of sequence
    "<|user_start|>",    # User message start
    "<|user_end|>",      # User message end
    "<|assistant_start|>", # Assistant message start
    "<|assistant_end|>", # Assistant message end
    "<|python_start|>",  # Python tool call start
    "<|python_end|>",    # Python tool call end
    "<|output_start|>",  # Tool output start
    "<|output_end|>",    # Tool output end
]
```

### Conversation Rendering

```python
def render_conversation(self, conversation, max_tokens=2048):
    """Convert conversation to token IDs with training mask."""
    ids, mask = [], []

    ids.append(bos); mask.append(0)

    for message in messages:
        if message["role"] == "user":
            ids += [user_start] + encode(content) + [user_end]
            mask += [0] * len(...)  # User tokens not supervised

        elif message["role"] == "assistant":
            ids.append(assistant_start); mask.append(0)
            if isinstance(content, str):
                ids += encode(content)
                mask += [1] * len(...)  # Assistant tokens supervised
            elif isinstance(content, list):
                for part in content:
                    if part["type"] == "python":
                        ids += [python_start] + encode(part["text"]) + [python_end]
                        mask += [1, 1, ..., 1]  # Tool calls supervised
                    elif part["type"] == "python_output":
                        ids += [output_start] + encode(part["text"]) + [output_end]
                        mask += [0, 0, ..., 0]  # Tool outputs not supervised
            ids.append(assistant_end); mask.append(1)

    return ids[:max_tokens], mask[:max_tokens]
```

---

## Data Pipeline

### Dataset Download

```python
# nanochat/dataset.py
def download_shards(num_shards):
    """Download FineWeb-Edu shards from HuggingFace."""
    for shard_id in range(num_shards):
        url = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2/resolve/main/data/CC-MAIN-2024-10/train-{shard_id:05d}.parquet"
        download_file_with_lock(url, f"shard_{shard_id:05d}.parquet")
```

### Distributed Data Loader

```python
def tokenizing_distributed_data_loader_bos_bestfit(tokenizer, batch_size, seq_len, split, device):
    """BOS-aligned dataloader with best-fit packing."""
    # Each rank processes different shards
    shard_indices = range(ddp_rank, num_shards, ddp_world_size)

    for shard_idx in shard_indices:
        # Load shard
        data = load_shard(shard_idx)

        # Tokenize documents
        for doc in data:
            doc_ids = tokenizer.encode(doc, prepend=bos)
            document_queue.append(doc_ids)

        # Pack into batches using best-fit
        while can_form_batch():
            batch = []
            for row_idx in range(batch_size):
                row = [bos]  # Start each row with BOS
                while len(row) < seq_len + 1:
                    # Best-fit: find largest document that fits
                    best_doc = find_best_fit(document_queue, seq_len - len(row))
                    if best_doc:
                        row.extend(best_doc)
                    else:
                        # Pad remainder
                        row.extend([bos] * (seq_len + 1 - len(row)))
                        break
                batch.append(row[:seq_len + 1])

            # Create tensors
            tensor = torch.tensor(batch, dtype=torch.long)
            inputs = tensor[:, :-1].to(device)
            targets = tensor[:, 1:].to(device)
            yield inputs, targets
```

---

## Optimization

### MuonAdamW Optimizer

```python
class MuonAdamW:
    """Combined optimizer: Muon for matrices, AdamW for embeddings."""

    def __init__(self, param_groups):
        for group in param_groups:
            if group['kind'] == 'muon':
                # Initialize Muon state
                for p in group['params']:
                    state = self.state[p]
                    state['momentum'] = torch.zeros_like(p)
            else:  # adamw
                # Initialize AdamW state
                for p in group['params']:
                    state = self.state[p]
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'muon':
                self._muon_step(group)
            else:
                self._adamw_step(group)

    def _muon_step(self, group):
        """Muon update with Newton-Schulz orthogonalization."""
        for p in group['params']:
            # Momentum update
            momentum = self.state[p]['momentum']
            momentum.mul_(group['momentum']).add_(p.grad)

            # Newton-Schulz iteration for orthogonalization
            X = momentum
            for _ in range(group['ns_steps']):
                A = X @ X.T
                X = 1.5 * X - 0.5 * A @ X

            # Weight decay
            if group['weight_decay'] > 0:
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

            # Update
            p.data.add_(X, alpha=-group['lr'])

    def _adamw_step(self, group):
        """Standard AdamW update."""
        for p in group['params']:
            grad = p.grad
            state = self.state[p]

            # Bias-corrected moment estimates
            state['exp_avg'].mul_(group['betas'][0]).add_(grad, alpha=1 - group['betas'][0])
            state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(grad, grad, value=1 - group['betas'][1])

            # AdamW update with weight decay
            p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
            p.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=-group['lr'])
```

### Learning Rate Scheduling

```python
def get_lr_multiplier(it, num_iterations, warmup_ratio, warmdown_ratio, final_lr_frac):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # Linear warmup
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # Constant LR
        return 1.0
    else:
        # Linear cooldown
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

---

## Distributed Training

### DDP Initialization

```python
def compute_init(device_type="cuda"):
    """Initialize distributed training if launched with torchrun."""
    ddp = is_ddp_requested()

    if ddp and device_type == "cuda":
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)

        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        device = torch.device(device_type)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

### Gradient Accumulation

```python
# In training loop
tokens_per_fwdbwd = batch_size * seq_len * world_size
grad_accum_steps = total_batch_size // tokens_per_fwdbwd

for step in range(num_iterations):
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        (loss / grad_accum_steps).backward()  # Gradients accumulate
        x, y, _ = next(train_loader)

    optimizer.step()
    model.zero_grad(set_to_none=True)
```

---

## Inference System

### KV Cache

```python
class KVCache:
    """Flash Attention 3 compatible KV cache."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        # Shape: (n_layers, B, T, H, D) - FA3 native layout
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, ...)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, ...)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, ...)

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """Clone cache for multi-sample generation."""
        pos = other.get_pos()
        self.k_cache[:, :, :pos] = other.k_cache[:, :, :pos]
        self.v_cache[:, :, :pos] = other.v_cache[:, :, :pos]
        self.cache_seqlens.fill_(pos)
```

### Token Sampling

```python
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample from logits with temperature and top-k."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None and top_k > 0:
        # Keep only top-k logits
        vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)
```

### Tool Use (Calculator)

```python
def use_calculator(expr):
    """Safely evaluate math expressions."""
    # Remove commas
    expr = expr.replace(",", "")

    # Pure math expressions
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # Disallow power
            return None
        return eval_with_timeout(expr)

    # String operations (e.g., "strawberry".count("r"))
    if '.count(' in expr:
        # Validate for safety
        dangerous = ['__', 'import', 'exec', 'eval', ...]
        if any(d in expr.lower() for d in dangerous):
            return None
        return eval_with_timeout(expr)

    return None
```

---

## Evaluation System

### CORE Benchmark

```python
def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """Evaluate on CORE benchmark (in-context learning tasks)."""
    results = {}

    for task in tasks:
        # Load task data
        data = load_task_data(task['dataset_uri'])

        # Evaluate accuracy
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        # Center by random baseline
        centered = (accuracy - baseline) / (1.0 - baseline)
        results[task['label']] = centered

    # CORE metric is mean centered accuracy
    core_metric = sum(results.values()) / len(results)
    return core_metric
```

### Chat Evaluation

```python
def run_generative_eval(task, tokenizer, model, engine, ...):
    """Evaluate generative tasks (free-form response)."""
    for i in range(num_problems):
        conversation = task[i]

        # Generate completion
        prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, ...)

        # Evaluate
        completions = [tokenizer.decode(r[len(prompt):]) for r in results]
        outcomes = [task.evaluate(conversation, c) for c in completions]
        passed = any(outcomes)  # Pass@k

        num_passed += int(passed)

    return num_passed / total

def run_categorical_eval(task, tokenizer, model, batch_size, ...):
    """Evaluate multiple-choice tasks (pick from A/B/C/D)."""
    for batch in batches:
        # Get logits for all conversations
        logits = model(prompt_ids)

        for idx, conversation in enumerate(batch):
            # Focus on answer position, only available letters
            letter_ids = [tokenizer.encode(l)[0] for l in conversation['letters']]
            focus_logits = logits[idx, answer_pos, letter_ids]

            # Predict by argmax
            predicted = conversation['letters'][focus_logits.argmax()]
            passed = task.evaluate(conversation, predicted)

    return num_passed / total
```

---

## Web Server Architecture

### Worker Pool

```python
class WorkerPool:
    """Pool of GPU workers for parallel inference."""

    def __init__(self, num_gpus):
        self.workers = []
        self.available_workers = asyncio.Queue()

    async def initialize(self, source, model_tag, step):
        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            model, tokenizer, _ = load_model(source, device, phase="eval", ...)
            engine = Engine(model, tokenizer)

            worker = Worker(gpu_id, device, engine, tokenizer, ...)
            self.workers.append(worker)
            await self.available_workers.put(worker)

    async def acquire_worker(self):
        return await self.available_workers.get()

    async def release_worker(self, worker):
        await self.available_workers.put(worker)
```

### Streaming Response

```python
async def generate_stream(worker, tokens, temperature, max_tokens, top_k):
    """Stream tokens as Server-Sent Events."""
    accumulated_tokens = []

    for token_column, _ in worker.engine.generate(tokens, ...):
        token = token_column[0]

        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        text = worker.tokenizer.decode(accumulated_tokens)

        # Only emit complete UTF-8 (no replacement chars)
        if not text.endswith('�'):
            new_text = text[len(last_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text})}\n\n"
            last_text = text

    yield f"data: {json.dumps({'done': True})}\n\n"
```

---

## Design Decisions

### Why These Choices?

| Decision | Rationale |
|----------|-----------|
| **Rotary embeddings** | Better length generalization than learned positions |
| **QK normalization** | Stabilizes attention for large models |
| **ReLU² activation** | Simpler than GELU, competitive performance |
| **No bias** | Reduces parameters, works well at scale |
| **Muon optimizer** | Better for matrix weights than AdamW alone |
| **BOS-aligned packing** | Each row starts fresh, cleaner training signal |
| **Sliding window** | Reduces memory for long sequences |
| **Flash Attention 3** | 2× faster than FA2 on Hopper GPUs |
| **tiktoken inference** | Fast, battle-tested tokenization |
| **rustbpe training** | Fast BPE training in Rust |

### Trade-offs

| Choice | Pro | Con |
|--------|-----|-----|
| Single codebase | Simple, hackable | Not production-ready |
| Fixed vocab 32K | Efficient | May hurt multilingual |
| Depth-based sizing | One dial to turn | Less control |
| FineWeb-Edu only | High quality | Limited diversity |

---

## Further Reading

- [Developer Guide](DEVELOPER_GUIDE.md) - How to modify the code
- [User Guide](USER_GUIDE.md) - How to use nanochat
- [Quick Start](QUICK_START.md) - Getting started quickly
- Original papers: RoPE, Flash Attention, Muon optimizer

---

*This guide is part of the nanochat documentation.*
