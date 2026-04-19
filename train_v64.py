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
DCLM v6.4 — Weg 3 Training (Checkpoint-Restart)
=================================================
Laedt den v6.3 Checkpoint (Step 17000) und trainiert
mit Neural Memory end-to-end weiter.

Voraussetzungen:
    pip install torch datasets tiktoken tqdm
    + dclm_v64.py im selben Ordner
    + checkpoints/dclm_v63_tinystories_best.pt vorhanden

Verwendung:
    python train_v64.py
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
    # v6.3 Checkpoint (Basis)
    "v63_checkpoint":  "checkpoints/dclm_v63_tinystories_best.pt",

    # Modell (muss zu v6.3 Checkpoint passen!)
    "hidden_dim":     768,
    "num_layers":     8,
    "num_heads":      12,
    "max_seq_len":    512,
    "dropout":        0.1,

    # NEU: Memory
    "use_memory":     True,
    "memory_heads":   (8, 9),    # Nur 2 Heads fuer Speed
    "mem_dim":        128,

    # Training — niedrigere LR weil wir feintunen
    "batch_size":     8,         # Kleiner wegen kein Checkpointing
    "grad_accum":     8,         # Effektiv immer noch 64
    "learning_rate":  1e-4,      # Niedriger als v6.3 (war 3e-4)
    "memory_lr":      3e-4,      # Memory darf schneller lernen
    "weight_decay":   0.1,
    "warmup_steps":   200,
    "max_steps":      10_000,
    "eval_interval":  250,
    "save_interval":  2500,
    "eval_tokens":    10,

    # fp16
    "use_fp16":       True,

    # Pfade
    "save_dir":       "checkpoints",
    "run_name":       "dclm_v64_tinystories",
}


# ============================================================
# DATASET (identisch zu v6.3 Training)
# ============================================================
class TinyStoriesDataset(Dataset):
    def __init__(self, split: str = "train", max_seq_len: int = 512):
        super().__init__()
        import tiktoken
        from datasets import load_dataset

        print(f"Lade TinyStories ({split})...")
        ds = load_dataset("roneneldan/TinyStories", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos = self.enc.eot_token
        self.max_seq_len = max_seq_len

        print("Tokenisiere...")
        all_tokens = []
        for example in tqdm(ds, desc="Tokenize"):
            tokens = self.enc.encode(example["text"], allowed_special=set())
            tokens.append(self.eos)
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = (len(self.tokens) - 1) // self.max_seq_len
        print(f"  {len(self.tokens):,} Tokens -> {self.num_samples:,} Samples "
              f"(seq_len={max_seq_len})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end   = start + self.max_seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


# ============================================================
# LEARNING RATE SCHEDULE
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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # --- Modell v6.4 erstellen ---
    from dclm_v64 import DCLM, DCLMConfig

    model_config = DCLMConfig(
        hidden_dim     = cfg["hidden_dim"],
        num_layers     = cfg["num_layers"],
        num_heads      = cfg["num_heads"],
        max_seq_len    = cfg["max_seq_len"],
        dropout        = cfg["dropout"],
        use_glu        = True,
        use_checkpoint = False,   # Aus wegen Memory-Loop
        use_head_gates = True,
        use_path_gates = True,
        # NEU v6.4:
        use_memory     = cfg["use_memory"],
        memory_heads   = cfg["memory_heads"],
        mem_dim        = cfg["mem_dim"],
    )

    model = DCLM(model_config).to(device)

    # --- v6.3 Checkpoint laden (Weg 3) ---
    print()
    missing, unexpected = model.load_v63_checkpoint(
        cfg["v63_checkpoint"], device=device
    )
    # Zeige welche Memory-Parameter neu sind
    mem_new = [k for k in missing if "memories" in k]
    print(f"  Davon Memory-Parameter: {len(mem_new)}")

    total_p = sum(p.numel() for p in model.parameters())
    mem_p   = sum(p.numel() for n, p in model.named_parameters() if "memories" in n)
    print(f"\nGesamt Params  : {total_p:,}")
    print(f"Memory Params  : {mem_p:,} ({mem_p/total_p*100:.1f}%)")
    print(f"Basis (v6.3)   : {total_p - mem_p:,}")

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

    # --- Optimizer mit getrennten LR-Gruppen ---
    # Memory-Parameter bekommen hoehere LR (muessen mehr lernen)
    # Basis-Parameter bekommen niedrigere LR (nur fein anpassen)
    memory_params = []
    base_decay_params = []
    base_no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "memories" in name:
            memory_params.append(p)
        elif p.dim() < 2 or "bias" in name or "norm" in name:
            base_no_decay_params.append(p)
        else:
            base_decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": base_decay_params,    "lr": cfg["learning_rate"],
         "weight_decay": cfg["weight_decay"]},
        {"params": base_no_decay_params, "lr": cfg["learning_rate"],
         "weight_decay": 0.0},
        {"params": memory_params,        "lr": cfg["memory_lr"],
         "weight_decay": 0.01},  # Wenig Decay fuer Memory
    ], betas=(0.9, 0.95))

    print(f"\nOptimizer-Gruppen:")
    print(f"  Basis (decay):    {len(base_decay_params)} Tensoren, lr={cfg['learning_rate']}")
    print(f"  Basis (no decay): {len(base_no_decay_params)} Tensoren, lr={cfg['learning_rate']}")
    print(f"  Memory:           {len(memory_params)} Tensoren, lr={cfg['memory_lr']}")

    # --- fp16 ---
    scaler = GradScaler("cuda", enabled=cfg["use_fp16"])

    # --- Checkpoint-Ordner ---
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # --- Training ---
    print(f"\nStart Training: {cfg['max_steps']} Steps")
    print(f"  Batch: {cfg['batch_size']} x {cfg['grad_accum']} = "
          f"{cfg['batch_size'] * cfg['grad_accum']} effektiv")
    print(f"  Base LR: {cfg['learning_rate']}, Memory LR: {cfg['memory_lr']}")
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

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR Schedule — getrennt fuer Base und Memory
        base_lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"],
                          cfg["learning_rate"])
        mem_lr  = get_lr(step, cfg["warmup_steps"], cfg["max_steps"],
                          cfg["memory_lr"])
        optimizer.param_groups[0]["lr"] = base_lr  # base decay
        optimizer.param_groups[1]["lr"] = base_lr  # base no decay
        optimizer.param_groups[2]["lr"] = mem_lr   # memory

        scaler.step(optimizer)
        scaler.update()
        step += 1

        # --- Logging ---
        if step % 50 == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = (50 * cfg["batch_size"] * cfg["grad_accum"]
                              * cfg["max_seq_len"]) / elapsed
            print(f"Step {step:6d} | loss {accum_loss/50:.4f} | "
                  f"blr {base_lr:.2e} mlr {mem_lr:.2e} | {tokens_per_sec:.0f} tok/s")
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

    # --- Final ---
    path = os.path.join(cfg["save_dir"], f"{cfg['run_name']}_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "config": model_config,
        "val_loss": best_val_loss,
    }, path)
    print(f"\nTraining fertig! Final: {path}")
    print(f"Best val_loss: {best_val_loss:.4f} (ppl={math.exp(best_val_loss):.1f})")

    # --- Generierungstest ---
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
