"""
eval_long_context.py — Context-Distance Scaling Evaluation
==========================================================
Trains *both* models on a simple copy-back task (learn to echo a target token
placed at a random early position when prompted at the end), then measures
retrieval accuracy as the sequence length (distance) grows.

This answers: "Does the O(1) hyperbolic state actually preserve information
               across longer distances than a standard Transformer?"

Usage:
    python eval_long_context.py [--train_steps 300] [--eval_trials 200]
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig

# Re-use the Transformer baseline from eval_baseline
from eval_baseline import TransformerBaseline


# ---------------------------------------------------------------------------
#  Copy-Back Task Generator
# ---------------------------------------------------------------------------
# Vocabulary layout:  0 = PAD,  1 = QUERY marker,  2..V-1 = content tokens
# Sequence:  [noise...] [TARGET] [noise...] [QUERY] → model must predict TARGET
VOCAB   = 512       # small vocab keeps training fast
N_TGTS  = 64        # how many distinct target values (2 .. N_TGTS+1)
Q_TOKEN = 1


def make_batch(batch_size: int, seq_len: int, device: torch.device):
    """
    Returns (input_ids, target_class).
    input_ids : (B, seq_len)    — filled with random noise, a TARGET early, QUERY at end
    target    : (B,)            — the TARGET token id the model should predict after QUERY
    """
    # fill with random noise tokens (avoid 0=PAD, 1=QUERY, use range 2..VOCAB-1)
    ids = torch.randint(N_TGTS + 2, VOCAB, (batch_size, seq_len), device=device)

    # place target in first quarter
    targets = torch.randint(2, N_TGTS + 2, (batch_size,), device=device)
    pos = torch.randint(1, max(2, seq_len // 4), (batch_size,), device=device)
    for b in range(batch_size):
        ids[b, pos[b]] = targets[b]

    # place QUERY at the very end
    ids[:, -1] = Q_TOKEN

    return ids, targets


# ---------------------------------------------------------------------------
#  Train a model on the copy-back task at a fixed short distance
# ---------------------------------------------------------------------------
def train_copy_task(tag: str, model: nn.Module, train_len: int,
                    steps: int, device: torch.device, use_bf16: bool):
    """
    Train the model to predict the TARGET token when it sees QUERY at position -1.
    We supervise only the last position to keep the task clean.
    """
    model.to(device)
    if use_bf16:
        model = model.bfloat16()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32

    print(f"\n  Training {tag} on copy-back (len={train_len}, {steps} steps) ...")
    t0 = time.perf_counter()

    for step in range(steps):
        ids, tgt = make_batch(batch_size, train_len, device)
        opt.zero_grad(set_to_none=True)

        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out   # (B, T, V)

        # supervise only the last-position logit → TARGET
        last_logits = logits[:, -1, :N_TGTS + 2]   # restrict to target vocab
        loss = loss_fn(last_logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, steps // 5) == 0:
            acc = (last_logits.argmax(-1) == tgt).float().mean().item() * 100
            print(f"    step {step:04d}  loss={loss.item():.4f}  acc={acc:.1f}%")

    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s")
    return model


# ---------------------------------------------------------------------------
#  Evaluate retrieval accuracy at increasing distances
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_distances(tag: str, model: nn.Module, distances: list[int],
                   trials: int, device: torch.device):
    model.eval()
    results = []

    print(f"\n  {'Distance':>8s}  {'Accuracy':>8s}  {'Correct':>8s}/{trials * 32}")
    for dist in distances:
        correct = 0
        total = 0
        for _ in range(trials):
            ids, tgt = make_batch(32, dist, device)
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits[:, -1, :N_TGTS + 2].argmax(-1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

        acc = correct / total * 100
        results.append({"distance": dist, "accuracy": round(acc, 2),
                        "correct": correct, "total": total})
        print(f"  {dist:>8d}  {acc:>7.1f}%  {correct:>8d}/{total}")

    return results


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_steps", type=int, default=300,
                    help="Steps to train each model on the copy-back task")
    ap.add_argument("--train_len", type=int, default=128,
                    help="Sequence length used during training phase")
    ap.add_argument("--eval_trials", type=int, default=50,
                    help="Number of batches per distance measurement")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_MODEL = 128     # smaller for speed — this is a retrieval probe, not LM quality
    N_LAYERS = 4
    MAX_SEQ = 4096

    distances = [64, 128, 256, 512, 1024, 2048]
    all_results = {}

    # ----- Transformer -----
    print("\n" + "=" * 60)
    print("  LONG-CONTEXT EVAL: Transformer Baseline")
    print("=" * 60)
    t_model = TransformerBaseline(VOCAB, D_MODEL, N_LAYERS, n_heads=4, max_seq=MAX_SEQ)
    t_model = train_copy_task("Transformer", t_model, args.train_len,
                              args.train_steps, device, use_bf16=False)
    # only test distances ≤ max_seq for transformer
    t_dists = [d for d in distances if d <= MAX_SEQ]
    all_results["transformer"] = eval_distances("Transformer", t_model, t_dists,
                                                 args.eval_trials, device)
    del t_model; torch.cuda.empty_cache()

    # ----- Hyper-SSM -----
    print("\n" + "=" * 60)
    print("  LONG-CONTEXT EVAL: Hyper-SSM")
    print("=" * 60)

    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = HyperSSM(HyperSSMConfig(VOCAB, D_MODEL, num_layers=N_LAYERS))
        def forward(self, x):
            return self.core(x, return_entropy=True)

    h_model = _Wrap()
    h_model = train_copy_task("Hyper-SSM", h_model, args.train_len,
                              args.train_steps, device, use_bf16=True)
    all_results["hyper_ssm"] = eval_distances("Hyper-SSM", h_model, distances,
                                               args.eval_trials, device)
    del h_model; torch.cuda.empty_cache()

    # ---- Save ----
    out = os.path.join(os.path.dirname(__file__), "results_long_context.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
