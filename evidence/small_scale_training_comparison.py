"""
Evidence Script: Small-Scale Training Comparison

This script sets up matched small training runs to compare:
- Baseline compressor (original naive style)
- Tiled compressor (our version)
- Optionally with/without Liquid Experts vs normal FFN

It is designed to be runnable even on modest hardware (CPU or small GPU)
as a template for larger experiments.

For real 100M–1B+ evidence, run this pattern on proper hardware with the
full training/train_hybrid_ultimate.py script using different --use_tiled flags
and ablations on the liquid MLP.

Current version: Very small model (few million params) for quick validation
of the comparison framework.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyper_ssm import HyperSSM, HyperSSMConfig
from hyper_ssm.tiled_compressor import TiledFractalCompressor
from hyper_ssm.hyperbolic_ops import FractalStateCompressor
import time

def create_small_model(use_tiled: bool = False, use_liquid: bool = True, hidden_size=128, num_layers=6):
    config = HyperSSMConfig(
        vocab_size=1000,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model = HyperSSM(
        config,
        use_hybrid=False,  # Keep simple for this comparison
        use_tiled_compressor=use_tiled,
    )
    # TODO: Add ablation for replacing liquid_mlp with standard FFN
    return model

def run_mini_training(model, steps=50, batch=4, seq_len=128, device="cpu"):
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    losses = []

    for step in range(steps):
        x = torch.randint(0, 1000, (batch, seq_len), device=device)
        y = torch.randint(0, 1000, (batch, seq_len), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 1000), y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    duration = time.time() - start
    return losses, duration

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running small-scale comparison on {device}")

    # Baseline (naive compressor)
    print("\n=== Baseline (Naive Compressor) ===")
    model_base = create_small_model(use_tiled=False, use_liquid=True)
    losses_base, t_base = run_mini_training(model_base, steps=30)

    # Tiled
    print("\n=== Tiled Compressor ===")
    model_tiled = create_small_model(use_tiled=True, use_liquid=True)
    losses_tiled, t_tiled = run_mini_training(model_tiled, steps=30)

    print("\n=== Summary ===")
    print(f"Baseline final loss: {losses_base[-1]:.4f}  (time: {t_base:.1f}s)")
    print(f"Tiled    final loss: {losses_tiled[-1]:.4f}  (time: {t_tiled:.1f}s)")

    # Note: For real evidence, scale this up dramatically and add proper ablations
    # (replace liquid_mlp with standard FFN, compare against real Mamba-2 blocks, etc.)
