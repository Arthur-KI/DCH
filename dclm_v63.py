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
DCLM v6.3c — Dynamic Gating (final, trainingsbereit)
===========================
Basis: dclm_v62_test.py (identische Kernarchitektur)
Neu:   Drei Gate-Erweiterungen fuer dynamisches Routing

1. Gated Residuals
   Jeder Block entscheidet per Gate wie stark Conv/State/FFN eingreifen.
   Statt: x = x + conv(x)
   Jetzt:  x = x + gate_c * conv(x)   (gate_c in [0,1] pro Token)

2. Per-Head Gates
   Nicht alle Heads laufen immer gleich stark.
   Jeder Head bekommt ein gelerntes Skalar-Gate pro Token.

3. Gated Head Mixing
   Das Modell lernt welche Skalen (Heads) gerade wichtig sind,
   statt alle Heads immer gleich zu gewichten.

Ergebnis: dynamisch geroutetes Mehrskalen-Modell
          statt starres Mehrskalen-System

Verwendung:
    from dclm_v63 import DCLM, DCLMConfig
    # oder direkt:
    python dclm_v63.py   (Testlauf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# CONFIG — Alle Einstellungen hier oben
# ============================================================
@dataclass
class DCLMConfig:
    # Tokenizer & Spezial-Tokens
    vocab_size:       int   = 50257
    eos_token_id:     int   = 50256  # GPT-2 EOS — Modell lernt hier aufzuhoeren
    pad_token_id:     int   = 50256  # Padding = EOS (wie bei GPT-2)
    # Architektur
    hidden_dim:       int   = 1024
    num_layers:       int   = 8
    num_heads:        int   = 12
    max_seq_len:      int   = 1024
    dropout:          float = 0.1
    base_kernel:      int   = 4
    dilations:        Tuple = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    use_glu:          bool  = True
    use_checkpoint:   bool  = True
    use_time_channel: bool  = True
    # Dynamic Gating (v6.3)
    use_head_gates:   bool  = True   # Per-Head Gates
    use_path_gates:   bool  = True   # Gated Residuals
    mix_gate_bottleneck: int = 0     # 0=full HxH, >0=bottleneck H->N->H (z.B. 64)


# ============================================================
# RMSNorm (ersetzt LayerNorm, wie in LLaMA)
# ============================================================
class RMSNorm(nn.Module):
    """Root Mean Square Norm — schneller als LayerNorm, kein Mean-Shift."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ============================================================
# TIME CHANNEL (identisch zu v6.2)
# ============================================================
class TimeChannel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim, bias=False)

    def forward(self, seq_len: int, device, dtype) -> torch.Tensor:
        pos      = torch.arange(seq_len, device=device, dtype=dtype)
        log_time = torch.log1p(pos) / 10.0
        sin_time = torch.sin(pos / 32.0)
        cos_time = torch.cos(pos / 32.0)
        signal   = torch.stack([log_time, sin_time, cos_time], dim=-1)
        return self.proj(signal).unsqueeze(0)


# ============================================================
# CORE MODULES (identisch zu v6.2)
# ============================================================
class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation    = dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=0, dilation=dilation, groups=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        return self.conv(F.pad(x, (pad, 0)))


class GLU(nn.Module):
    def __init__(self, hidden_dim: int, expand_factor: int = 4):
        super().__init__()
        inner        = hidden_dim * expand_factor
        self.proj_in  = nn.Linear(hidden_dim, inner * 2, bias=False)
        self.proj_out = nn.Linear(inner, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        content, gate = h.chunk(2, dim=-1)
        return self.proj_out(content * torch.sigmoid(gate))


class DilatedConvStack(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 4,
                 dilations: Tuple = (1, 2, 4)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(DilatedCausalConv1d(hidden_dim, kernel_size, d), nn.GELU())
            for d in dilations
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        for layer in self.layers:
            h = h + layer(h)
        return self.proj(h.transpose(1, 2))


# ============================================================
# NEU: Multi-Head Dilated State MIT Per-Head Gates
# ============================================================
class MultiHeadDilatedState(nn.Module):
    """
    Identisch zu v6.2 PLUS optionale Per-Head Gates.

    Per-Head Gates (NEU):
    Jeder Head bekommt ein gelerntes Skalar-Gate [0,1] pro Token.
    Das Modell kann Heads situativ an- oder abschalten:
      head_weights = sigmoid(head_router(x))  # [B, S, num_heads]
      out = sum(head_weights[:,:,i] * head_i(x))
    """
    def __init__(self, hidden_dim: int, num_heads: int = 12,
                 kernel_size: int = 4, use_glu: bool = True,
                 use_head_gates: bool = True,
                 mix_gate_bottleneck: int = 0):
        super().__init__()
        assert hidden_dim % num_heads == 0,             f"hidden_dim ({hidden_dim}) muss durch num_heads ({num_heads}) teilbar sein!"
        self.hidden_dim    = hidden_dim
        self.num_heads     = num_heads
        self.head_dim      = hidden_dim // num_heads
        self.use_glu       = use_glu
        self.use_head_gates      = use_head_gates
        self.mix_gate_bottleneck = mix_gate_bottleneck

        # Exakt dieselben Patterns wie v6.2
        dilation_patterns = [
            (1, 2, 4),        # Head 0: Standard Grammatik
            (1, 1, 1),        # Head 1: Ultra-Lokal
            (4, 8, 16),       # Head 2: Phrase
            (8, 16, 32),      # Head 3: Halber Satz
            (32, 64, 128),    # Head 4: Paragraph
            (64, 128, 256),   # Head 5: Seite
            (256, 512, 1024), # Head 6: Kapitel
            (1, 100, 200),    # Head 7: Mittel-Sprung
            (1, 500, 1000),   # Head 8: Weit-Sprung
            (1, 1024, 2048),  # Head 9: Extrem-Sprung
            (3, 9, 27),       # Head 10: Ungerade
            (5, 25, 125),     # Head 11: 5er Schritte
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

        # NEU: Per-Head Router
        # Lernt pro Token welche Heads relevant sind
        # Init nahe 0 -> sigmoid nahe 0.5 -> alle Heads gleich am Anfang
        if use_head_gates:
            self.head_router = nn.Linear(hidden_dim, num_heads, bias=True)
            nn.init.zeros_(self.head_router.weight)
            nn.init.zeros_(self.head_router.bias)

        # Post-Mix Gate: filtert nach dem Head-Concat (immer aktiv wenn head_gates)
        # Varianten: full (HxH, teuer) oder bottleneck (H->B->H, guenstiger)
        self.mix_gate_bottleneck = mix_gate_bottleneck
        if use_head_gates:
            bn = self.mix_gate_bottleneck
            if bn > 0:
                # Bottleneck: H -> bn -> H (viel weniger Parameter)
                self.mix_gate = nn.Sequential(
                    nn.Linear(hidden_dim, bn, bias=False),
                    nn.GELU(),
                    nn.Linear(bn, hidden_dim, bias=True),
                )
                nn.init.zeros_(self.mix_gate[-1].weight)
                nn.init.constant_(self.mix_gate[-1].bias, 2.0)
            else:
                # Full: HxH Gate
                self.mix_gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
                nn.init.zeros_(self.mix_gate.weight)
                nn.init.constant_(self.mix_gate.bias, 2.0)  # sigmoid(2)≈0.88

        self.mixing = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape

        # x ist hier bereits der vom Block normalisierte State-Input (norm2)
        if self.use_head_gates:
            # [B, S, num_heads] in [0, 1]
            head_weights = torch.sigmoid(self.head_router(x))

        if self.use_glu:
            g = self.gate_proj(x)
            x_gated, gate = g.chunk(2, dim=-1)
            x_gated = x_gated * torch.sigmoid(gate)
        else:
            x_gated = x

        x_heads = x_gated.view(B, S, self.num_heads, self.head_dim)

        head_outputs = []
        for i, head_convs in enumerate(self.heads):
            h = x_heads[:, :, i, :].transpose(1, 2)  # [B, head_dim, S]
            for conv in head_convs:
                h = h + conv(h)
            h = h.transpose(1, 2)  # [B, S, head_dim]

            # NEU: Per-Head Gate anwenden
            if self.use_head_gates:
                w = head_weights[:, :, i:i+1]  # [B, S, 1]
                h = h * w

            head_outputs.append(h)

        out = torch.cat(head_outputs, dim=-1)  # [B, S, H]

        # NEU: Post-Mix Gate — Modell kann nach dem Concat nochmal filtern
        if self.use_head_gates:
            out = out * torch.sigmoid(self.mix_gate(out))

        return self.mixing(out)


# ============================================================
# NEU: DCLM Block MIT Gated Residuals
# ============================================================
class DCLMBlock(nn.Module):
    """
    Identisch zu v6.2 PLUS Gated Residuals.

    Gated Residuals (NEU):
    Jeder Pfad (Conv, State, FFN) bekommt ein Gate das entscheidet
    wie stark dieser Pfad eingreift:

    v6.2:  x = x + dropout(conv(norm(x)))
    v6.3:  x = x + gate_c * dropout(conv(norm(x)))

    gate_c = sigmoid(linear(x))  in [0,1] pro Token
    Init nahe 1 -> Verhalten am Anfang wie v6.2
    """
    def __init__(self, config: DCLMConfig):
        super().__init__()
        H = config.hidden_dim

        self.norm1 = RMSNorm(H)
        self.norm2 = RMSNorm(H)
        self.norm3 = RMSNorm(H)

        self.conv  = DilatedConvStack(H, config.base_kernel, config.dilations[:6])
        self.state = MultiHeadDilatedState(
            H, config.num_heads, config.base_kernel, config.use_glu,
            use_head_gates=config.use_head_gates,
            mix_gate_bottleneck=getattr(config, 'mix_gate_bottleneck', 0),
        )
        self.ffn     = GLU(H)
        self.dropout = nn.Dropout(config.dropout)

        # NEU: Path Gates — getrennte Gates pro Pfad fuer bessere Entkopplung
        # Berechnung aus normalisiertem Input (stabiler als rohes x)
        # Init-Bias = 2.0 -> sigmoid(2) ≈ 0.88 -> anfangs wie v6.2
        self.use_path_gates = config.use_path_gates
        if config.use_path_gates:
            self.conv_gate  = nn.Linear(H, 1, bias=True)
            self.state_gate = nn.Linear(H, 1, bias=True)
            self.ffn_gate   = nn.Linear(H, 1, bias=True)
            for g in [self.conv_gate, self.state_gate, self.ffn_gate]:
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_path_gates:
            # Normalisierte Inputs (stabiler fuer Gate-Berechnung)
            n1 = self.norm1(x)
            n2 = self.norm2(x)
            n3 = self.norm3(x)
            # Gates aus normalisiertem Input — getrennt und entkoppelt
            gc = torch.sigmoid(self.conv_gate(n1))   # [B, S, 1]
            gs = torch.sigmoid(self.state_gate(n2))  # [B, S, 1]
            gf = torch.sigmoid(self.ffn_gate(n3))    # [B, S, 1]
            x = x + gc * self.dropout(self.conv(n1))
            x = x + gs * self.dropout(self.state(n2))
            x = x + gf * self.dropout(self.ffn(n3))
        else:
            # Identisch zu v6.2
            x = x + self.dropout(self.conv(self.norm1(x)))
            x = x + self.dropout(self.state(self.norm2(x)))
            x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# ============================================================
# HAUPTMODELL (identisch zu v6.2 bis auf Block-Typ)
# ============================================================
class DCLM(nn.Module):
    """
    DCLM v6.3c — Dynamic Gating (final, trainingsbereit)

    API-kompatibel zu v6.2.
    Checkpoints von v6.2 sind NICHT direkt ladbar (neue Parameter),
    aber die Architektur ist eine echte Obermenge.
    """
    def __init__(self, config: DCLMConfig):
        super().__init__()
        self.config = config
        self.embed  = nn.Embedding(config.vocab_size, config.hidden_dim)

        if config.use_time_channel:
            self.time_channel = TimeChannel(config.hidden_dim)
        else:
            self.time_channel = None

        self.blocks     = nn.ModuleList([DCLMBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)
        self.dropout    = nn.Dropout(config.dropout)

        self.apply(self._init_weights)
        # Gate-Inits NACH self.apply() setzen — sonst werden sie ueberschrieben!
        self._init_gate_biases()

    def _init_gate_biases(self):
        """Stellt Gate-Inits nach self.apply() wieder her."""
        for block in self.blocks:
            if getattr(block, 'use_path_gates', False):
                for g in [block.conv_gate, block.state_gate, block.ffn_gate]:
                    nn.init.zeros_(g.weight)
                    nn.init.constant_(g.bias, 2.0)
            state = block.state
            if getattr(state, 'use_head_gates', False):
                nn.init.zeros_(state.head_router.weight)
                nn.init.zeros_(state.head_router.bias)
                # mix_gate kann Linear oder Sequential sein (Bottleneck)
                if isinstance(state.mix_gate, nn.Linear):
                    nn.init.zeros_(state.mix_gate.weight)
                    nn.init.constant_(state.mix_gate.bias, 2.0)
                else:
                    nn.init.zeros_(state.mix_gate[-1].weight)
                    nn.init.constant_(state.mix_gate[-1].bias, 2.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, S = input_ids.shape
        device = input_ids.device

        if S > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len:]
            S = input_ids.shape[1]

        x = self.embed(input_ids)

        if self.time_channel is not None:
            x = x + self.time_channel(S, device, x.dtype)

        x = self.dropout(x)

        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x      = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # Weight tying

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {'logits': logits, 'loss': loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 repetition_penalty: float = 1.2) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx    = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx)['logits'][:, -1] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    prev = input_ids[b, -100:].unique()
                    logits[b, prev] /= repetition_penalty

            if top_k > 0:
                topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            probs      = F.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)

            # Bei EOS aufhoeren
            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids


# ============================================================
# PARAMETER-ANALYSE
# ============================================================
def count_new_params(config: DCLMConfig) -> dict:
    """Zeigt wie viele Parameter die neuen Gates hinzufuegen."""
    H = config.hidden_dim
    L = config.num_layers
    N = config.num_heads

    path_gate_params  = (H * 1 + 1) * 3 * L if config.use_path_gates else 0
    head_router_params = (H * N + N) * L      if config.use_head_gates else 0
    bn = getattr(config, 'mix_gate_bottleneck', 0)
    if bn > 0:
        mix_gate_params = (H * bn + bn * H + H) * L if config.use_head_gates else 0
    else:
        mix_gate_params = (H * H + H) * L           if config.use_head_gates else 0
    head_gate_params   = head_router_params + mix_gate_params

    return {
        'path_gate_params':   path_gate_params,
        'head_router_params': head_router_params,
        'mix_gate_params':    mix_gate_params,
        'head_gate_params':   head_gate_params,
        'total_new':          path_gate_params + head_gate_params,
    }


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("DCLM v6.3c — Dynamic Gating (final, trainingsbereit) Test")
    print("=" * 50)

    config_small = DCLMConfig(
        vocab_size     = 50257,
        hidden_dim     = 768,
        num_layers     = 8,
        num_heads      = 12,
        max_seq_len    = 1024,
        dropout        = 0.1,
        use_glu        = True,
        use_checkpoint = False,  # Fuer Test aus
        use_head_gates = True,
        use_path_gates = True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = DCLM(config_small).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    new_p   = count_new_params(config_small)

    print(f"Device         : {device}")
    print(f"Hidden         : {config_small.hidden_dim}")
    print(f"Layers         : {config_small.num_layers}")
    print(f"Heads          : {config_small.num_heads}")
    print(f"")
    print(f"Gesamt Params  : {total_p:,}")
    print(f"Neue Gate-P    : {new_p['total_new']:,}  "
          f"({new_p['total_new']/total_p*100:.2f}%)")
    print(f"  Path Gates   : {new_p['path_gate_params']:,}")
    print(f"  Head Router  : {new_p['head_router_params']:,}")
    print(f"  Mix Gate     : {new_p['mix_gate_params']:,}")
    print(f"  Head Gates   : {new_p['head_gate_params']:,}")
    print()

    # Forward test
    x = torch.randint(0, 50257, (2, 128)).to(device)
    y = torch.randint(0, 50257, (2, 128)).to(device)

    with torch.no_grad():
        out = model(x, y)

    print(f"Forward OK     : loss={out['loss']:.4f}  "
          f"logits={tuple(out['logits'].shape)}")

    # Gate-Aktivierungen inspizieren
    print()
    print("Gate-Aktivierungen (erstes Block, random Input):")
    block0 = model.blocks[0]
    xn = model.embed(x)
    if model.time_channel is not None:
        xn = xn + model.time_channel(128, device, model.embed.weight.dtype)
    with torch.no_grad():
        if config_small.use_path_gates:
            n1 = block0.norm1(xn); n2 = block0.norm2(xn); n3 = block0.norm3(xn)
            gc = torch.sigmoid(block0.conv_gate(n1))
            gs = torch.sigmoid(block0.state_gate(n2))
            gf = torch.sigmoid(block0.ffn_gate(n3))
            print(f"  Path Gates (mean): "
                  f"conv={gc.mean():.3f}  "
                  f"state={gs.mean():.3f}  "
                  f"ffn={gf.mean():.3f}")
        if config_small.use_head_gates:
            n2_test = block0.norm2(xn)
            hw = torch.sigmoid(block0.state.head_router(n2_test))
            print(f"  Head Gates (mean per head): "
                  + "  ".join(f"H{i}={hw[:,:,i].mean():.3f}"
                               for i in range(config_small.num_heads)))
    print()
    print("v6.3 bereit!")