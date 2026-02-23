"""
DCH - Dilated Convolutional Heads
=================================
- Keine Positional Embeddings
- TimeChannel: 3 Features (log + sin + cos)
- Spezialisierte Dilated Convolutional Heads
- GLU FFN
- LLaMA Tokenizer (TinyLlama)
- EOS Token am Ende jedes Textes
- Auto-Balanced Dataset

Trainierbar mit Mixed Dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import math
import random

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# ============================================================
# CONFIG
# ============================================================

@dataclass  
class DCHConfig:
    vocab_size: int = 32000         # LLaMA Tokenizer
    hidden_dim: int = 384
    num_layers: int = 6
    num_heads: int = 8             
    max_seq_len: int = 1024
    dropout: float = 0.1
    base_kernel: int = 4
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    use_glu: bool = True
    use_checkpoint: bool = True
    use_time_channel: bool = True


@dataclass
class DataMixConfig:
    """Einstellbare Dataset-Gewichtung"""
    tiny_stories: float = 0.30
    gutenberg: float = 0.25
    wikipedia: float = 0.25
    conflict_memory: float = 0.20
    
    # Automatische Normalisierung
    def __post_init__(self):
        total = self.tiny_stories + self.gutenberg + self.wikipedia + self.conflict_memory
        if total > 0:
            self.tiny_stories /= total
            self.gutenberg /= total
            self.wikipedia /= total
            self.conflict_memory /= total


# ============================================================
# TIME CHANNEL
# ============================================================

class TimeChannel(nn.Module):
    """
    Einfacher Time-Channel mit 3 Features:
    - log(t+1): Monotone Ordnung (früher vs später)
    - sin(t/32): Periodische Feinstruktur
    - cos(t/32): Periodische Feinstruktur (phasenverschoben)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim, bias=False)
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        
        log_time = torch.log1p(pos) / 10.0
        sin_time = torch.sin(pos / 32.0)
        cos_time = torch.cos(pos / 32.0)
        
        time_signal = torch.stack([log_time, sin_time, cos_time], dim=-1)
        return self.proj(time_signal).unsqueeze(0)


# ============================================================
# CORE MODULES
# ============================================================

class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        return self.conv(x)


class GLU(nn.Module):
    def __init__(self, hidden_dim: int, expand_factor: int = 4):
        super().__init__()
        inner_dim = hidden_dim * expand_factor
        self.proj_in = nn.Linear(hidden_dim, inner_dim * 2, bias=False)
        self.proj_out = nn.Linear(inner_dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        content, gate = h.chunk(2, dim=-1)
        h = content * torch.sigmoid(gate)
        return self.proj_out(h)


class DilatedConvStack(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 4, dilations: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                DilatedCausalConv1d(hidden_dim, kernel_size, dilation=d),
                nn.GELU(),
            )
            for d in dilations
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        for layer in self.layers:
            h = h + layer(h)
        h = h.transpose(1, 2)
        return self.proj(h)


class MultiHeadDilatedState(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, kernel_size: int = 4, use_glu: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_glu = use_glu
        
        dilation_patterns = [
            (1, 2, 4),
            (4, 8, 16),
            (16, 32, 64),
            (64, 128, 256),
            (256, 512, 1024),
            (1, 16, 256),
            (4, 64, 1024),
            (16, 256, 2048),
        ]
        
        self.heads = nn.ModuleList()
        for dilations in dilation_patterns[:num_heads]:
            head = nn.ModuleList([
                DilatedCausalConv1d(self.head_dim, kernel_size, d)
                for d in dilations
            ])
            self.heads.append(head)
        
        if use_glu:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        
        self.mixing = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        
        if self.use_glu:
            g = self.gate_proj(x)
            x_gated, gate = g.chunk(2, dim=-1)
            x_gated = x_gated * torch.sigmoid(gate)
        else:
            x_gated = x
        
        x_heads = x_gated.view(B, S, self.num_heads, self.head_dim)
        
        head_outputs = []
        for i, head_convs in enumerate(self.heads):
            h = x_heads[:, :, i, :].transpose(1, 2)
            for conv in head_convs:
                h = h + conv(h)
            head_outputs.append(h.transpose(1, 2))
        
        out = torch.cat(head_outputs, dim=-1)
        return self.mixing(out)


class DCHBlock(nn.Module):
    def __init__(self, config: DCHConfig):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        
        self.conv = DilatedConvStack(config.hidden_dim, config.base_kernel, config.dilations[:6])
        self.state = MultiHeadDilatedState(config.hidden_dim, config.num_heads, config.base_kernel, config.use_glu)
        self.ffn = GLU(config.hidden_dim, expand_factor=4)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.conv(self.norm1(x)))
        x = x + self.dropout(self.state(self.norm2(x)))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# ============================================================
# MODEL
# ============================================================

class DCH(nn.Module):
    """DCH v6.2 + TimeChannel"""
    
    def __init__(self, config: DCHConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.time_channel = TimeChannel(config.hidden_dim) if config.use_time_channel else None
        self.blocks = nn.ModuleList([DCHBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        B, S = input_ids.shape
        device = input_ids.device
        
        if S > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len:]
            B, S = input_ids.shape
        
        x = self.embed(input_ids)
        
        if self.time_channel is not None:
            x = x + self.time_channel(S, device, x.dtype)
        
        x = self.dropout(x)
        
        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 repetition_penalty: float = 1.2) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx)['logits'][:, -1] / max(temperature, 1e-6)
            
            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    prev = input_ids[b, -100:].unique()
                    logits[b, prev] /= repetition_penalty
            
            if top_k > 0:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float('-inf')
            
            probs = F.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================
# BALANCED DATASET
# ============================================================

class BalancedMixDataset(Dataset):
    """
    Auto-balanced Dataset mit EOS Token.
    Samplet proportional aus verschiedenen Quellen.
    """
    def __init__(
        self, 
        tokenizer, 
        max_seq_len: int = 1024,
        total_samples: int = 50000,
        mix_config: DataMixConfig = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mix_config = mix_config or DataMixConfig()
        
        # EOS Token
        self.eos_token = tokenizer.eos_token or "</s>"
        self.pad_token_id = tokenizer.pad_token_id
        
        # Lade alle Quellen
        print("📚 Loading datasets...")
        self.sources: Dict[str, List[str]] = {}
        
        # TinyStories
        if self.mix_config.tiny_stories > 0:
            n = int(total_samples * self.mix_config.tiny_stories)
            self.sources['tiny_stories'] = self._load_tinystories(n)
        
        # Gutenberg
        if self.mix_config.gutenberg > 0:
            n = int(total_samples * self.mix_config.gutenberg)
            self.sources['gutenberg'] = self._load_gutenberg(n)
        
        # Wikipedia
        if self.mix_config.wikipedia > 0:
            n = int(total_samples * self.mix_config.wikipedia)
            self.sources['wikipedia'] = self._load_wikipedia(n)
        
        # Conflict/Memory Tasks
        if self.mix_config.conflict_memory > 0:
            n = int(total_samples * self.mix_config.conflict_memory)
            self.sources['conflict'] = self._generate_conflict_data(n)
        
        # Flatten mit Balancing
        self.samples = self._balance_and_flatten()
        print(f"✅ Total balanced samples: {len(self.samples)}")
        
    def _load_tinystories(self, n: int) -> List[str]:
        try:
            from datasets import load_dataset
            print(f"  Loading TinyStories ({n})...")
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            texts = []
            for sample in ds:
                if len(texts) >= n:
                    break
                text = sample['text'].strip()
                if len(text) > 50:
                    texts.append(text)
            print(f"  ✓ TinyStories: {len(texts)}")
            return texts
        except Exception as e:
            print(f"  ⚠ TinyStories failed: {e}")
            return []
    
    def _load_gutenberg(self, n: int) -> List[str]:
        try:
            from datasets import load_dataset
            print(f"  Loading Gutenberg ({n})...")
            ds = load_dataset("pg19", split="train", streaming=True)
            texts = []
            for sample in ds:
                if len(texts) >= n:
                    break
                # Chunke lange Texte
                full_text = sample['text']
                chunks = [full_text[i:i+2000] for i in range(0, min(len(full_text), 20000), 2000)]
                for chunk in chunks:
                    if len(chunk) > 100 and len(texts) < n:
                        texts.append(chunk.strip())
            print(f"  ✓ Gutenberg: {len(texts)}")
            return texts
        except Exception as e:
            print(f"  ⚠ Gutenberg failed: {e}")
            return []
    
    def _load_wikipedia(self, n: int) -> List[str]:
        try:
            from datasets import load_dataset
            print(f"  Loading Wikipedia ({n})...")
            ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            texts = []
            for sample in ds:
                if len(texts) >= n:
                    break
                text = sample['text'][:2000].strip()
                if len(text) > 100:
                    texts.append(text)
            print(f"  ✓ Wikipedia: {len(texts)}")
            return texts
        except Exception as e:
            print(f"  ⚠ Wikipedia failed: {e}")
            return []
    
    def _generate_conflict_data(self, n: int) -> List[str]:
        """Generiert Memory/Conflict Tasks"""
        print(f"  Generating Conflict data ({n})...")
        
        templates = [
            # Farben
            ("{name}'s {obj} was {attr1}. Later it changed to {attr2}. {name}'s original {obj} was {attr1}.",
             ["red", "blue", "green", "yellow", "black", "white"],
             ["ball", "car", "hat", "book", "bag", "key"]),
            
            # Orte
            ("{name} lived in {attr1}. Then {name} moved to {attr2}. {name}'s hometown was {attr1}.",
             ["Berlin", "Paris", "London", "Tokyo", "Rome", "Madrid"],
             None),
            
            # Zahlen
            ("The {obj} cost {attr1} dollars. The price changed to {attr2} dollars. The original price was {attr1}.",
             ["10", "20", "50", "100", "200", "500"],
             ["book", "lamp", "chair", "phone", "watch", "bag"]),
            
            # Namen
            ("{name}'s pet was named {attr1}. Later called {attr2}. The original name was {attr1}.",
             ["Luna", "Max", "Bella", "Charlie", "Lucy", "Oscar"],
             None),
        ]
        
        names = ["Tim", "Emma", "Tom", "Lisa", "Max", "Anna", "Ben", "Sara"]
        
        texts = []
        for _ in range(n):
            template, attrs, objs = random.choice(templates)
            attr1, attr2 = random.sample(attrs, 2)
            name = random.choice(names)
            
            if objs:
                obj = random.choice(objs)
                text = template.format(name=name, obj=obj, attr1=attr1, attr2=attr2)
            else:
                text = template.format(name=name, attr1=attr1, attr2=attr2)
            
            # Manchmal Filler hinzufügen
            if random.random() > 0.5:
                fillers = [
                    "The sun was shining. ",
                    "It was a beautiful day. ",
                    "Time passed quickly. ",
                    "Many things happened. ",
                ]
                filler = "".join(random.choices(fillers, k=random.randint(1, 5)))
                parts = text.split(". ", 1)
                if len(parts) == 2:
                    text = parts[0] + ". " + filler + parts[1]
            
            texts.append(text)
        
        print(f"  ✓ Conflict: {len(texts)}")
        return texts
    
    def _balance_and_flatten(self) -> List[Tuple[str, str]]:
        """Balanciert und merged alle Quellen"""
        all_samples = []
        
        for source_name, texts in self.sources.items():
            for text in texts:
                all_samples.append((source_name, text))
        
        random.shuffle(all_samples)
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        source, text = self.samples[idx]
        
        # Fallback
        if not text or len(text.strip()) < 10:
            text = "Once upon a time there was a story."
        
        # EOS Token anhängen!
        text = text.strip() + self.eos_token
        
        # Tokenize
        enc = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_seq_len,
            padding='max_length', 
            return_tensors='pt'
        )
        
        input_ids = enc['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        return {'input_ids': input_ids, 'labels': labels, 'source': source}


# ============================================================
# TRAINING
# ============================================================

def load_tokenizer():
    """Lädt LLaMA Tokenizer (TinyLlama)"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def main():
    tokenizer = load_tokenizer()
    print(f"✅ Tokenizer: TinyLlama (vocab={tokenizer.vocab_size})")
    
    config = DCHConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=384,
        num_layers=6,
        num_heads=8,
        max_seq_len=1024,
        dropout=0.1,
        use_glu=True,
        use_checkpoint=True,
        use_time_channel=True
    )
    
    # Dataset Mix (einstellbar!)
    mix_config = DataMixConfig(
        tiny_stories=0.30,      # 30% Kindergeschichten
        gutenberg=0.25,         # 25% Klassiker
        wikipedia=0.25,         # 25% Fakten
        conflict_memory=0.20    # 20% Memory Tasks
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("DCH + TIME CHANNEL + BALANCED DATA")
    print("="*70)
    print(f"Device: {device}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Layers: {config.num_layers}")
    print(f"Context: {config.max_seq_len}")
    print(f"TimeChannel: ✅")
    print(f"EOS Token: ✅ '{tokenizer.eos_token}'")
    print(f"\nData Mix:")
    print(f"  TinyStories:  {mix_config.tiny_stories:.0%}")
    print(f"  Gutenberg:    {mix_config.gutenberg:.0%}")
    print(f"  Wikipedia:    {mix_config.wikipedia:.0%}")
    print(f"  Conflict:     {mix_config.conflict_memory:.0%}")
    print()
    
    model = DCH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Dataset
    full_dataset = BalancedMixDataset(
        tokenizer,
        max_seq_len=config.max_seq_len,
        total_samples=50000,
        mix_config=mix_config
    )
    
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    if HAS_BNB:
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        print("Using 8-bit Adam ✓")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*3)
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    NUM_EPOCHS = 3
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        model.train()
        total_loss, n = 0, 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(dtype=torch.float16):
                loss = model(x, y)['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            n += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'avg': f'{total_loss/n:.3f}'})
            
            if i > 0 and i % 2000 == 0:
                model.eval()
                print(f"\n\n--- Sample bei Step {i} ---")
                for prompt in ["Once upon a time", "Berlin is", "The original color was"]:
                    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
                    text = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"[{prompt}] → {text}\n")
                model.train()
        
        train_loss = total_loss / n
        
        # Validation
        model.eval()
        val_loss_total, val_n = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                y = batch['labels'].to(device)
                with autocast(dtype=torch.float16):
                    loss = model(x, y)['loss']
                val_loss_total += loss.item()
                val_n += 1
        val_loss = val_loss_total / val_n
        
        gap = val_loss - train_loss
        print(f"\n📊 Epoch {epoch}:")
        print(f"   Train Loss: {train_loss:.4f} (PPL: {math.exp(min(train_loss, 10)):.2f})")
        print(f"   Val Loss:   {val_loss:.4f} (PPL: {math.exp(min(val_loss, 10)):.2f})")
        print(f"   Gap:        {gap:+.4f}")
        
        # Samples
        print(f"\n--- Samples Epoch {epoch} ---")
        for prompt in ["Once upon a time", "The capital of France is", "Tim's original key was"]:
            ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=70, temperature=0.8, top_k=40)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[{prompt}] → {text}\n")
        
        # Checkpoint
        torch.save({
            'config': config,
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tokenizer_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'mix_config': mix_config
        }, f'dch_epoch{epoch}.pt')
        print(f"✓ Checkpoint: dch_epoch{epoch}.pt")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Final Val Loss: {val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
