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
DCLM — Text Generation from Checkpoint
=======================================
Loads a checkpoint and generates text. Works with v6.3 and v6.4.

Usage:
    python generate.py
    python generate.py --model v64 --checkpoint checkpoints/dclm_v64_tinystories_best.pt
    python generate.py --prompt "The little cat" --tokens 200 --temp 0.9
"""

import argparse
import torch
import tiktoken


def generate():
    parser = argparse.ArgumentParser(description="DCLM Text Generation")
    parser.add_argument("--model", type=str, default="v64",
                        choices=["v63", "v64"],
                        help="Model version (v63 or v64)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if not set)")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    parser.add_argument("--num", type=int, default=3,
                        help="Number of generations")
    args = parser.parse_args()

    # Auto-detect checkpoint path
    if args.checkpoint is None:
        if args.model == "v64":
            args.checkpoint = "checkpoints/dclm_v64_tinystories_best.pt"
        else:
            args.checkpoint = "checkpoints/dclm_v63_tinystories_best.pt"

    # Import correct model version
    if args.model == "v64":
        from dclm_v64 import DCLM
    else:
        from dclm_v63 import DCLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  Step: {step}  |  Val-Loss: {val_loss}")

    model = DCLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Tokens: {args.tokens}  |  Temp: {args.temp}  |  Top-k: {args.top_k}")
    print("=" * 60)

    tokens = enc.encode(args.prompt)
    input_ids = torch.tensor([tokens], device=device)

    for i in range(args.num):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k,
                repetition_penalty=args.rep_penalty,
            )
        text = enc.decode(output[0].tolist())
        eos_marker = enc.decode([config.eos_token_id])
        if eos_marker in text:
            text = text[:text.index(eos_marker)]
        print(f"\n--- Generation {i+1} ---")
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    generate()
