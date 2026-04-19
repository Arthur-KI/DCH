# DCLM — Dilated Convolution Language Model

A **convolution-only** language model architecture with Dynamic Gating and Neural Memory. No attention mechanism — achieves **Perplexity 3.0** on TinyStories with only 120M parameters and linear memory scaling.

## Key Features

- **No Attention** — Entirely based on dilated causal convolutions, no quadratic memory cost
- **Multi-Scale Heads** — 12 heads with different dilation patterns, from ultra-local (1,1,1) to extreme long-range (1,1024,2048)
- **Dynamic Gating (v6.3)** — Per-head gates, path gates, and gated head mixing for learned routing
- **Neural Memory (v6.4)** — Titan-inspired fast-weight memory in long-range heads, trained end-to-end
- **RMSNorm** — LLaMA-style normalization for faster training
- **Linear Scaling** — Memory usage scales linearly with sequence length, not quadratically

## Results on TinyStories

Trained on the complete [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (474M tokens).

| Version | Steps | Val Loss | Perplexity | Parameters |
|---------|-------|----------|------------|------------|
| v6.3    | 500   | 2.62     | 13.8       | 119.2M     |
| v6.3    | 5,000 | 1.42     | 4.2        | 119.2M     |
| v6.3    | 10,000| 1.27     | 3.6        | 119.2M     |
| v6.3    | 17,000| 1.18     | 3.3        | 119.2M     |
| v6.4    | +250  | 1.15     | 3.2        | 119.6M     |
| v6.4    | +10,000| **1.08** | **3.0**    | 119.6M     |

The Neural Memory (v6.4) adds only 0.3% more parameters but improves val loss by 0.096 — extremely parameter-efficient.

## Architecture

```
Input → Embedding → TimeChannel → [DCLMBlock × N] → RMSNorm → Output

DCLMBlock:
  x → norm1 → DilatedConvStack  → gate_conv  × dropout → + residual
  x → norm2 → MultiHeadState    → gate_state × dropout → + residual
  x → norm3 → GLU (FFN)         → gate_ffn   × dropout → + residual
```

### Multi-Scale Heads (12 heads, each with different dilation patterns)

| Head | Dilation Pattern | Role |
|------|-----------------|------|
| 0 | (1, 2, 4) | Standard grammar |
| 1 | (1, 1, 1) | Ultra-local |
| 2 | (4, 8, 16) | Phrase |
| 3 | (8, 16, 32) | Half-sentence |
| 4 | (32, 64, 128) | Paragraph |
| 5 | (64, 128, 256) | Page |
| 6 | (256, 512, 1024) | Chapter |
| 7 | (1, 100, 200) | Medium-jump |
| 8 | (1, 500, 1000) | Long-jump + Memory |
| 9 | (1, 1024, 2048) | Extreme-jump + Memory |
| 10 | (3, 9, 27) | Odd patterns |
| 11 | (5, 25, 125) | Powers of 5 |

### Dynamic Gating (v6.3)

Three gating mechanisms for learned routing:

1. **Path Gates** — Each path (Conv/State/FFN) has a per-token gate [0,1] that controls its contribution
2. **Per-Head Gates** — Each head gets a learned scalar gate per token, allowing heads to be dynamically activated/deactivated
3. **Gated Head Mixing** — Post-concatenation gate for feature-level filtering

All gates are initialized near 1.0 (bias=2.0, sigmoid≈0.88) so the model starts behaving like an ungated version and gradually learns when to reduce paths.

### Neural Memory (v6.4, Titan-inspired)

Long-range heads (configurable, default: 8, 9) get a fast-weight memory module:

- **READ**: Query vector retrieves from memory matrix → past information
- **WRITE**: Key-value pairs write new knowledge into memory
- **FORGET**: Sigmoid gate controls how strongly new knowledge overwrites old

The memory uses chunk-based processing (stride=64) for efficient GPU utilization. Memory is initialized passively (bias=-2.0, sigmoid≈0.12) to not disrupt pretrained weights.

## Quick Start

### Installation

```bash
pip install torch datasets tiktoken tqdm
```

### Test the model

```bash
# v6.3 (Dynamic Gating)
python dclm_v63.py

# v6.4 (+ Neural Memory)
python dclm_v64.py
```

### Train on TinyStories

```bash
# Train v6.3 from scratch
python train_tinystories.py

# Train v6.4 with Memory (requires v6.3 checkpoint)
python train_v64.py
```

### Generate text

```bash
# From v6.3 checkpoint
python generate.py

# From v6.4 checkpoint
python generate.py --checkpoint checkpoints/dclm_v64_tinystories_best.pt --model v64

# Custom prompt
python generate.py --prompt "The little dog" --tokens 200 --temp 0.9
```

## Configuration

All settings are at the top of each file. Key parameters in `DCLMConfig`:

```python
DCLMConfig(
    vocab_size     = 50257,       # GPT-2 vocabulary
    hidden_dim     = 768,         # Model dimension
    num_layers     = 8,           # Number of DCLM blocks
    num_heads      = 12,          # Multi-scale heads
    max_seq_len    = 1024,        # Maximum sequence length
    dropout        = 0.1,
    # Dynamic Gating (v6.3)
    use_head_gates = True,        # Per-head gates
    use_path_gates = True,        # Gated residuals
    # Neural Memory (v6.4)
    use_memory     = True,        # Enable neural memory
    memory_heads   = (8, 9),      # Which heads get memory
    mem_dim        = 128,         # Memory value dimension
)
```

## Checkpoint Compatibility

v6.3 checkpoints can be loaded into v6.4 models:

```python
from dclm_v64 import DCLM, DCLMConfig

config = DCLMConfig(use_memory=True, memory_heads=(8, 9))
model = DCLM(config)
model.load_v63_checkpoint("checkpoints/dclm_v63_tinystories_best.pt")
# All v6.3 weights loaded, memory parameters freshly initialized
# Continue training end-to-end (Way 3)
```

## Sample Generations (v6.4, Step 10000, PPL 3.0)

> Once upon a time there was a little girl named Lola. She was three years old and she loved adventure. One day, Lola decided to go on an adventure. She walked through the woods until she came across a very tall stack of lumber. Lola couldn't believe her eyes! It looked so special just looking out from far away. She wanted a closer look, so she climbed up the pile of lumber and started climbing it. When Lola reached the top, she saw a mysterious...

## File Overview

| File | Description |
|------|-------------|
| `dclm_v63.py` | v6.3 model — Dynamic Gating |
| `dclm_v64.py` | v6.4 model — + Neural Memory |
| `train_tinystories.py` | Training script for v6.3 on TinyStories |
| `train_v64.py` | Finetuning script for v6.4 (loads v6.3 checkpoint) |
| `generate.py` | Text generation from checkpoints |
| `LICENSE` | Apache License 2.0 |

## Design Philosophy

Traditional language models rely on attention for token interactions. DCLM takes a different approach:

1. **Fixed multi-scale structure** provides robust, diverse receptive fields via dilated convolutions
2. **Dynamic gating** lets the model learn when each scale is relevant
3. **Neural memory** adds adaptive long-term storage where fixed convolutions fall short

The result is a model that combines the efficiency of convolutions (linear memory) with the adaptivity of learned routing and explicit memory — without attention.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
