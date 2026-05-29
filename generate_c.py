"""
generate_c.py — Interactive C code generation from trained Hyper-SSM
=====================================================================
Loads a checkpoint and generates embedded C code from a prompt.

Usage:
    python generate_c.py                             # interactive mode
    python generate_c.py --prompt "#include <stdio.h>"
    python generate_c.py --prompt "void uart_init(" --out generated.c
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    config = HyperSSMConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    )
    model = HyperSSM(config).to(device).bfloat16()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[MODEL] Loaded from {ckpt_path}")
    print(f"        Best training loss: {ckpt.get('best_loss', 'N/A')}")
    print(f"        Total steps trained: {ckpt.get('total_steps', 'N/A')}")
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 300,
             temperature: float = 0.7, top_p: float = 0.9,
             device: str = "cuda") -> str:
    ids = tokenizer.encode(prompt)["input_ids"].to(device)

    for _ in range(max_tokens):
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        if torch.isnan(logits).any():
            print(f"\\n[FATAL] NaNs detected in logits at step {_}!")
            break
            
        # Cast to fp32 immediately to prevent bfloat16 underflow in softmax
        logits = logits[:, -1, :].to(torch.float32) / temperature

        # nucleus (top-p) sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
        remove = cum - torch.softmax(sorted_logits, -1) >= top_p
        sorted_logits[remove] = -float("inf")
        probs = torch.softmax(sorted_logits, -1)
        tok = sorted_idx.gather(-1, torch.multinomial(probs, 1))
        ids = torch.cat([ids, tok], dim=1)

        if tok.item() == tokenizer.tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hyper_ssm_c_code.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="Write output to .c file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HyperTokenizer()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        print("        Run: python train_c_code.py")
        return

    model = load_model(args.ckpt, device)

    if args.prompt:
        # single-shot mode
        result = generate(model, tokenizer, args.prompt,
                          args.max_tokens, args.temperature, args.top_p,
                          str(device))
        print("\n" + result)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"\n[SAVED] {args.out}")
    else:
        # interactive REPL
        print("\n[INTERACTIVE MODE] Type C code to continue. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input("C> ")
                if not prompt.strip():
                    continue
                result = generate(model, tokenizer, prompt,
                                  args.max_tokens, args.temperature, args.top_p,
                                  str(device))
                print(result)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
