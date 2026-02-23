# DCH - Dilated Convolutional Heads

Ein experimentelles Language Model mit **spezialisierten Dilated Convolutional Heads** - ohne Positional Embeddings.

## ✨ Features

- **Keine Attention** - O(n) statt O(n²) Komplexität
- **Keine Positional Embeddings** - nur TimeChannel
- **TimeChannel** - Explizites Zeitgefühl durch log + sin/cos
- **Spezialisierte Heads** - Jeder Head hat fixe Reichweite (erzwungene Spezialisierung)
- **GLU (Gated Linear Units)** - Bessere Feature-Selektion

## 🧠 Architektur

```
Input → Embedding → TimeChannel → [DCHBlock × N] → Output

DCHBlock:
├── DilatedConvStack (lokale Patterns)
├── MultiHeadDilatedState (spezialisierte Heads)
└── GLU FFN (Feature-Mixing)
```

### Spezialisierte Heads (Kernidee)

Im Gegensatz zu Transformer-Heads, die alle das gleiche lernen können, hat jeder DCH-Head ein **fixes Dilation-Pattern**:

| Head | Dilations | Spezialisierung |
|------|-----------|-----------------|
| 1 | (1, 2, 4) | Grammatik, Wortebene |
| 2 | (4, 8, 16) | Phrasen |
| 3 | (16, 32, 64) | Sätze |
| 4 | (64, 128, 256) | Paragraphen |
| 5 | (256, 512, 1024) | Long-Range, Kapitel |
| 6 | (1, 16, 256) | Multi-Scale |
| 7 | (4, 64, 1024) | Multi-Scale |
| 8 | (16, 256, 2048) | Ultra Long-Range |

**Heads können nicht gleich werden** - sie sind strukturell verschieden!

### Keine Positional Embeddings

DCH nutzt **keine** klassischen Positional Embeddings. Stattdessen:

**TimeChannel** mit 3 Features:
```python
log_time = log(position + 1)    # Monoton: früher vs später
sin_time = sin(position / 32)   # Periodische Feinstruktur  
cos_time = cos(position / 32)   # Phasenverschoben
```

Warum?
- Positional Embeddings = "Token X ist an Position 42"
- TimeChannel = "Token X kam früher/später als Token Y"

Das gibt echtes **Zeitgefühl** statt nur Position.

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/DCH.git
cd DCH
pip install -r requirements.txt
```

## 🚀 Training

```bash
python dch.py
```

### Dataset-Mix anpassen

```python
mix_config = DataMixConfig(
    tiny_stories=0.30,      # Kindergeschichten
    gutenberg=0.25,         # Klassische Literatur
    wikipedia=0.25,         # Fakten
    conflict_memory=0.20    # Memory Tasks
)
```

## 🧪 Testen

Nach dem Training:

```bash
python test_model.py --checkpoint dch_epoch3.pt
```

Interaktiv:

```bash
python test_model.py --checkpoint dch_epoch3.pt --interactive
```

## 📊 Modell-Konfiguration

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `hidden_dim` | 384 | Embedding-Dimension |
| `num_layers` | 6 | Anzahl DCHBlocks |
| `num_heads` | 8 | Anzahl spezialisierter Heads |
| `max_seq_len` | 1024 | Maximale Sequenzlänge |
| `base_kernel` | 4 | Kernel-Größe |

### Größere Konfigurationen

```python
# ~55M Parameter
config = DCHConfig(
    hidden_dim=512,
    num_layers=8,
    num_heads=8
)

# ~120M Parameter
config = DCHConfig(
    hidden_dim=768,
    num_layers=12,
    num_heads=12
)
```

## 📈 Ergebnisse

Das Modell lernt erfolgreich:
- ✅ Zeitliche Abhängigkeiten (ohne Positional Embeddings!)
- ✅ "Original vs Aktuell" Unterscheidung
- ✅ Kohärente Textgenerierung

## 🔬 Vergleich

| Feature | Transformer | DCH |
|---------|-------------|-----|
| Komplexität | O(n²) | O(n) |
| Positional Embedding | Ja | ❌ Nein |
| TimeChannel | Nein | ✅ Ja |
| Heads | Können gleich werden | Strukturell verschieden |
| VRAM | Viel | Weniger |

## 📁 Dateien

```
DCH/
├── dch.py              # Modell + Training
├── test_model.py       # Testen & Inference
├── requirements.txt    # Dependencies
├── README.md           # Diese Datei
└── LICENSE             # MIT License
```

## 🙏 Inspirationen

- Dilated Convolutions: WaveNet
- GLU: "Language Modeling with Gated Convolutional Networks"
- TimeChannel: Fourier Features

## 📄 Lizenz

MIT License - siehe [Arthur-Ki](2026)

---

**Experimentelles Projekt** - Feedback willkommen! 🚀
