# Copyright 2025-2026 Arthur
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DCLM v6.3 — TinyStories Training
=================================
- TinyStories komplett (HuggingFace Download)
- GPT-2 Tokenizer (vocab_size=50257, EOS=50256)
- fp16 Mixed Precision
- cudnn.benchmark fuer schnelle Convolutions
- Checkpoint-Speicherung

Voraussetzungen (pip install):
    pip install torch datasets tiktoken tqdm

Verwendung:
    python train_tinystories.py
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================
# TRAINING CONFIG — Alles Einstellbare hier oben
# ============================================================
TRAIN_CONFIG = {
    # Modell
    "hidden_dim":     768,
    "num_layers":     8,
    "num_heads":      12,
    "max_seq_len":    512,     # TinyStories sind kurz, 512 reicht
    "dropout":        0.1,

    # Training
    "batch_size":     16,       # Anpassen je nach VRAM (8 bei 8GB, 16 bei 12GB+)
    "grad_accum":     4,        # Effektive Batch = batch_size * grad_accum = 64
    "learning_rate":  3e-4,
    "weight_decay":   0.1,
    "warmup_steps":   500,
    "max_steps":      50_000,   # Fuer komplett: ~100k+, fuer Test: 5000
    "eval_interval":  500,
    "save_interval":  5000,
    "eval_tokens":    10,       # Anzahl Eval-Batches

    # fp16
    "use_fp16":       True,

    # Pfade
    "save_dir":       "checkpoints",
    "run_name":       "dclm_v63_tinystories",
}


# ============================================================
# DATASET
# ============================================================
class TinyStoriesDataset(Dataset):
    """Laedt TinyStories komplett, tokenisiert mit GPT-2, haengt EOS an."""

    def __init__(self, split: str = "train", max_seq_len: int = 512):
        super().__init__()
        import tiktoken
        from datasets import load_dataset

        print(f"Lade TinyStories ({split})...")
        ds = load_dataset("roneneldan/TinyStories", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos = self.enc.eot_token  # 50256
        self.max_seq_len = max_seq_len

        # Alles tokenisieren und in einen langen Strom packen
        print("Tokenisiere...")
        all_tokens = []
        for example in tqdm(ds, desc="Tokenize"):
            tokens = self.enc.encode(example["text"], allowed_special=set())
            tokens.append(self.eos)  # EOS nach jeder Story
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = (len(self.tokens) - 1) // self.max_seq_len
        print(f"  {len(self.tokens):,} Tokens → {self.num_samples:,} Samples "
              f"(seq_len={max_seq_len})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end   = start + self.max_seq_len + 1  # +1 fuer Label-Shift
        chunk = self.tokens[start:end]
        x = chunk[:-1]   # Input
        y = chunk[1:]     # Target (shifted by 1)
        return x, y


# ============================================================
# LEARNING RATE SCHEDULE (Cosine mit Warmup)
# ============================================================
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    cfg = TRAIN_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Schnelle Convolutions auf CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # --- Modell ---
    from dclm_v63 import DCLM, DCLMConfig

    model_config = DCLMConfig(
        hidden_dim     = cfg["hidden_dim"],
        num_layers     = cfg["num_layers"],
        num_heads      = cfg["num_heads"],
        max_seq_len    = cfg["max_seq_len"],
        dropout        = cfg["dropout"],
        use_glu        = True,
        use_checkpoint = True,
        use_head_gates = True,
        use_path_gates = True,
    )

    model = DCLM(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {total_params:,} Parameter")

    # --- Dataset ---
    train_ds = TinyStoriesDataset("train", cfg["max_seq_len"])
    val_ds   = TinyStoriesDataset("validation", cfg["max_seq_len"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True, drop_last=True,
    )

    # --- Optimizer ---
    # Kein Weight Decay auf Bias, Norm, Gate-Bias
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["learning_rate"], betas=(0.9, 0.95))

    # --- fp16 ---
    scaler = GradScaler("cuda", enabled=cfg["use_fp16"])

    # --- Checkpoint-Ordner ---
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # --- Training ---
    print(f"\nStart Training: {cfg['max_steps']} Steps")
    print(f"  Batch: {cfg['batch_size']} x {cfg['grad_accum']} = "
          f"{cfg['batch_size'] * cfg['grad_accum']} effektiv")
    print(f"  fp16: {cfg['use_fp16']}")
    print()

    model.train()
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    accum_loss = 0.0
    t_start = time.time()

    while step < cfg["max_steps"]:
        optimizer.zero_grad()

        # Gradient Accumulation
        for micro in range(cfg["grad_accum"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with autocast("cuda", enabled=cfg["use_fp16"]):
                out = model(x, y)
                loss = out["loss"] / cfg["grad_accum"]

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR Schedule
        lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"],
                     cfg["learning_rate"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()
        step += 1

        # --- Logging ---
        if step % 50 == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = (50 * cfg["batch_size"] * cfg["grad_accum"]
                              * cfg["max_seq_len"]) / elapsed
            print(f"Step {step:6d} | loss {accum_loss/50:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            accum_loss = 0.0
            t_start = time.time()

        # --- Eval ---
        if step % cfg["eval_interval"] == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with autocast("cuda", enabled=cfg["use_fp16"]):
                        vout = model(vx, vy)
                    val_loss += vout["loss"].item()
                    val_batches += 1
                    if val_batches >= cfg["eval_tokens"]:
                        break

            val_loss /= val_batches
            print(f"  >>> EVAL Step {step}: val_loss={val_loss:.4f} "
                  f"(ppl={math.exp(val_loss):.1f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = os.path.join(cfg["save_dir"],
                                    f"{cfg['run_name']}_best.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": model_config,
                    "val_loss": val_loss,
                }, path)
                print(f"  >>> Neuer Best! Gespeichert: {path}")

            model.train()

        # --- Checkpoint ---
        if step % cfg["save_interval"] == 0:
            path = os.path.join(cfg["save_dir"],
                                f"{cfg['run_name']}_step{step}.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_config,
                "val_loss": best_val_loss,
            }, path)
            print(f"  Checkpoint: {path}")

    # --- Fertiges Modell speichern ---
    path = os.path.join(cfg["save_dir"], f"{cfg['run_name']}_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "config": model_config,
        "val_loss": best_val_loss,
    }, path)
    print(f"\nTraining fertig! Final: {path}")
    print(f"Best val_loss: {best_val_loss:.4f} (ppl={math.exp(best_val_loss):.1f})")

    # --- Kurzer Generierungstest ---
    print("\n--- Generierungstest ---")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt = "Once upon a time"
    tokens = enc.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    output = model.generate(input_ids, max_new_tokens=100,
                            temperature=0.8, top_k=50)
    print(enc.decode(output[0].tolist()))


if __name__ == "__main__":
    train()
