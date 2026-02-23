"""
DCH Test Script
===============
Testen von trainierten DCH Modellen.

Usage:
    python test_model.py --checkpoint dch_epoch3.pt
    python test_model.py --checkpoint dch_epoch3.pt --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer

# Import Model
from dch import DCH, DCHConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Lädt Modell und Tokenizer aus Checkpoint."""
    print(f"📂 Lade: {checkpoint_path}")
    
    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = chk['config']
    model = DCH(config).to(device)
    model.load_state_dict(chk['state_dict'])
    model.eval()
    
    # Tokenizer
    tokenizer_name = chk.get('tokenizer_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modell geladen | {num_params/1e6:.1f}M Parameter")
    print(f"✅ Tokenizer: {tokenizer_name}")
    print(f"✅ Epoch: {chk.get('epoch', '?')} | Val Loss: {chk.get('val_loss', '?'):.4f}")
    
    return model, tokenizer, config


def generate_text(model, tokenizer, prompt: str, device: str = 'cuda',
                  max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int = 50, repetition_penalty: float = 1.2) -> str:
    """Generiert Text aus einem Prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


def test_memory(model, tokenizer, prompt: str, target: str, distractor: str, device: str = 'cuda'):
    """Testet ob Modell target > distractor vorhersagt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    distractor_ids = tokenizer.encode(distractor, add_special_tokens=False)
    
    with torch.no_grad():
        logits = model(input_ids)['logits'][0, -1]
        
        score_target = logits[target_ids[0]].item()
        score_distractor = logits[distractor_ids[0]].item()
    
    diff = score_target - score_distractor
    return score_target, score_distractor, diff


def run_memory_tests(model, tokenizer, config, device: str = 'cuda'):
    """Führt Memory-Tests mit verschiedenen Filler-Längen durch."""
    print("\n" + "="*70)
    print("MEMORY TEST")
    print("="*70)
    
    filler_base = "The sun was shining brightly. Birds were singing. Everything was peaceful. "
    
    print(f"\n{'Filler':<10} | {'Tokens':<8} | {'Target':<10} | {'Distract':<10} | {'Diff':<10} | Result")
    print("-"*70)
    
    for filler_mult in [0, 5, 10, 20, 30, 40, 50, 60]:
        filler = filler_base * filler_mult
        
        prompt = f"Tim had a red key. {filler}Later Tim saw a blue key. Tim's original key was"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        total_tokens = input_ids.shape[1]
        
        if total_tokens > config.max_seq_len:
            print(f"{filler_mult:<10} | {total_tokens:<8} | OUT OF CONTEXT")
            continue
        
        score_red, score_blue, diff = test_memory(model, tokenizer, prompt, "red", "blue", device)
        result = "✅ RED" if diff > 0 else "❌ BLUE"
        
        print(f"{filler_mult:<10} | {total_tokens:<8} | {score_red:<10.2f} | {score_blue:<10.2f} | {diff:<+10.2f} | {result}")
    
    print("="*70)


def run_generalization_tests(model, tokenizer, device: str = 'cuda'):
    """Testet Generalisierung auf andere Attribute."""
    print("\n" + "="*70)
    print("GENERALISIERUNG TEST")
    print("="*70)
    
    tests = [
        ("Tim lived in Berlin. Later Tim moved to Paris. Tim's hometown was", "Berlin", "Paris"),
        ("The book cost 50 dollars. The price changed to 80 dollars. The original price was", "50", "80"),
        ("Emma's cat was named Luna. She started calling it Star. The cat's original name was", "Luna", "Star"),
        ("The house was big. After renovation it became small. Originally the house was", "big", "small"),
    ]
    
    print(f"\n{'Test':<60} | {'Diff':<8} | Result")
    print("-"*80)
    
    correct = 0
    for prompt, target, distractor in tests:
        score_t, score_d, diff = test_memory(model, tokenizer, prompt, target, distractor, device)
        result = "✅" if diff > 0 else "❌"
        if diff > 0:
            correct += 1
        
        short_prompt = prompt[:55] + "..." if len(prompt) > 55 else prompt
        print(f"{short_prompt:<60} | {diff:<+8.2f} | {result}")
    
    print("="*80)
    print(f"Ergebnis: {correct}/{len(tests)} korrekt")


def interactive_mode(model, tokenizer, device: str = 'cuda'):
    """Interaktiver Modus für Textgenerierung."""
    print("\n" + "="*70)
    print("INTERAKTIVER MODUS")
    print("="*70)
    print("Eingabe: Text-Prompt")
    print("Commands: /quit, /temp <float>, /topk <int>, /tokens <int>")
    print("="*70 + "\n")
    
    temperature = 0.8
    top_k = 50
    max_tokens = 100
    
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break
        
        if not prompt:
            continue
        
        if prompt == "/quit":
            print("Bye! 👋")
            break
        elif prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature: {temperature}")
            except:
                print("Usage: /temp <float>")
            continue
        elif prompt.startswith("/topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top-k: {top_k}")
            except:
                print("Usage: /topk <int>")
            continue
        elif prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens: {max_tokens}")
            except:
                print("Usage: /tokens <int>")
            continue
        
        output = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
        print(f"\n{output}\n")


def main():
    parser = argparse.ArgumentParser(description='DCH Test Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Pfad zum Checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true', help='Interaktiver Modus')
    parser.add_argument('--skip-memory', action='store_true', help='Memory-Tests überspringen')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Device: {device}")
    
    model, tokenizer, config = load_model(args.checkpoint, device)
    
    if not args.skip_memory:
        run_memory_tests(model, tokenizer, config, device)
        run_generalization_tests(model, tokenizer, device)
    
    # Sample Generation
    print("\n" + "="*70)
    print("SAMPLE GENERATION")
    print("="*70)
    
    prompts = [
        "Once upon a time",
        "The scientist discovered",
        "In a dark forest",
    ]
    
    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, device, max_new_tokens=80)
        print(f"\n[{prompt}]")
        print(f"→ {output}")
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
