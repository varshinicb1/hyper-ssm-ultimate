"""
train_c_code.py — Train Hyper-SSM on real embedded C source code
=================================================================
Loads the assembled corpus from data/c_corpus.txt, tokenizes with GPT-2 BPE,
and trains the full architecture with bfloat16 + CUDA Riemannian kernels.

Usage:
    python train_c_code.py [--epochs 3] [--batch 4] [--seq_len 512] [--lr 3e-4]
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer

# ---------- Centralized CUDA kernel acceleration (2026 refactor) ----------
from hyper_ssm.cuda_ops import (
    is_cuda_available,
    get_lorentz_product,
    get_project_to_tangent,
)

_lorentz_product = get_lorentz_product()
_project_to_tangent = get_project_to_tangent()

def lorentz_product(x, y):
    """Lorentz inner product (CUDA when available, else high-quality PyTorch)."""
    return _lorentz_product(x, y)

def project_to_tangent(x, g):
    """Riemannian projection of gradients onto tangent space (uses CUDA if present)."""
    return _project_to_tangent(x, g)

if is_cuda_available():
    print("[SYSTEM] Using accelerated CUDA Riemannian kernels via hyper_ssm.cuda_ops")
else:
    print("[SYSTEM] Using pure PyTorch Riemannian ops (CUDA extension not loaded)")


# ===================================================================
#  Data loader — streams real C code as overlapping token windows
# ===================================================================
class CCodeDataLoader:
    """
    Reads the pre-built C corpus, tokenizes once, then yields
    overlapping (X, Y) windows for next-token prediction.
    """

    def __init__(self, corpus_path: str, tokenizer, batch_size: int,
                 seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len

        print(f"[DATA] Loading {corpus_path} ...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[DATA] Corpus: {len(text):,} chars ({len(text)/1e6:.2f} MB)")

        # tokenize entire corpus once
        enc = tokenizer.encode(text)
        self.tokens = enc["input_ids"][0].tolist()
        print(f"[DATA] Tokens: {len(self.tokens):,}")

    def num_batches(self, epochs: int = 1) -> int:
        usable = len(self.tokens) - self.seq_len - 1
        per_epoch = max(1, usable // (self.batch_size * self.seq_len))
        return per_epoch * epochs

    def stream(self, epochs: int = 1):
        """Yield (X, Y) batches, cycling through the corpus `epochs` times."""
        stride = self.seq_len  # non-overlapping for clean coverage
        for _epoch in range(epochs):
            ptr = 0
            while True:
                xs, ys = [], []
                for _ in range(self.batch_size):
                    end = ptr + self.seq_len + 1
                    if end > len(self.tokens):
                        ptr = 0
                        end = self.seq_len + 1
                    chunk = self.tokens[ptr:end]
                    xs.append(chunk[:-1])
                    ys.append(chunk[1:])
                    ptr += stride
                yield (torch.tensor(xs, dtype=torch.long),
                       torch.tensor(ys, dtype=torch.long))
                # stop epoch when we've swept the corpus
                if ptr + self.seq_len + 1 > len(self.tokens):
                    break


# ===================================================================
#  Generation helper
# ===================================================================
@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_tokens: int = 200,
                    temperature: float = 0.8, top_p: float = 0.9,
                    device: str = "cuda"):
    """Generate C code continuation from a prompt."""
    model.eval()
    ids = tokenizer.encode(prompt)["input_ids"].to(device)

    for _ in range(max_tokens):
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :] / temperature

        # nucleus sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
        remove = cum_probs - torch.softmax(sorted_logits, -1) >= top_p
        sorted_logits[remove] = -float("inf")
        probs = torch.softmax(sorted_logits, -1)
        next_tok = sorted_idx.gather(-1, torch.multinomial(probs, 1))
        ids = torch.cat([ids, next_tok], dim=1)

        # stop at eof marker or newline overrun
        if next_tok.item() == tokenizer.tokenizer.eos_token_id:
            break

    model.train()
    return tokenizer.decode(ids[0])


# ===================================================================
#  Training loop
# ===================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    tokenizer = HyperTokenizer()
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "c_corpus.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus not found at {corpus_path}")
        print("        Run: python data/prepare_c_corpus.py")
        return

    loader = CCodeDataLoader(corpus_path, tokenizer, args.batch, args.seq_len)

    config = HyperSSMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=12,
    )
    model = HyperSSM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {params / 1e6:.2f}M parameters")

    # Train in fp32 to prevent epsilon underflows that cause NaNs
    torch.set_float32_matmul_precision("medium")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_batches = loader.num_batches(args.epochs)
    print(f"[TRAIN] {args.epochs} epochs, ~{total_batches} batches, "
          f"seq_len={args.seq_len}, batch={args.batch}\n")

    history = []
    best_loss = float("inf")
    t0 = time.perf_counter()

    try:
        model.train()
        for step, (X, Y) in enumerate(loader.stream(args.epochs)):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, entropy = model(X, return_entropy=True)
            loss = criterion(logits.reshape(-1, config.vocab_size), Y.reshape(-1))

            if torch.isnan(loss):
                print(f"\\n[FATAL] NaN at step {step}. Aborting.")
                break

            total_loss = loss - 0.01 * entropy
            total_loss.backward()

            # Riemannian gradient projection for hyperbolic params
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None and ("W_state.weight" in name or "W_input.weight" in name):
                        p.grad = project_to_tangent(p, p.grad)
                # Zero any NaN gradients across ALL parameters
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        p.grad.zero_()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            entry = {"step": step, "loss": round(loss.item(), 4)}
            history.append(entry)

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % 20 == 0:
                elapsed = time.perf_counter() - t0
                toks = (step + 1) * args.batch * args.seq_len
                tps = toks / elapsed if elapsed > 0 else 0
                print(f"  Step {step:05d}  loss={loss.item():.4f}  "
                      f"best={best_loss:.4f}  entropy={entropy.item():.3f}  "
                      f"{tps:.0f} tok/s")

            # Log every 100 steps for overnight monitoring
            if step > 0 and step % 100 == 0:
                elapsed = time.perf_counter() - t0
                eta_hrs = (total_batches - step) * (elapsed / step) / 3600
                print(f"  [ETA] ~{eta_hrs:.1f}h remaining")
    except KeyboardInterrupt:
        print("\\n[STOP] Keyboard interrupt detected. Saving checkpoint...")

    # save checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "hyper_ssm_c_code.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": config.vocab_size,
                   "hidden_size": config.hidden_size,
                   "num_layers": config.num_layers},
        "best_loss": best_loss,
        "total_steps": len(history),
    }, ckpt_path)
    print(f"\n[SAVE] Checkpoint → {ckpt_path}  (best loss: {best_loss:.4f})")

    # save training history
    hist_path = os.path.join(os.path.dirname(__file__), "results_c_training.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"[SAVE] History  → {hist_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    train(ap.parse_args())


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
