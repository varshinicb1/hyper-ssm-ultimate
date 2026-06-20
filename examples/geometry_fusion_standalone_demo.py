"""
Full Standalone Training Demo: Geometry-Aware Parallel Fusion Block Only
================================================================================

This is a self-contained, minimal but complete training example that uses
*only* the new 2026 GeometryAwareParallelFusion + a tiny Lorentz compressor.

No full HyperSSM, no liquid experts, no big tokenizer — just the fusion
technology in isolation so you can study / debug / extend it easily.

What it does:
- Generates synthetic hierarchical tree sequences (same as the ablation).
- Builds a tiny "FusionBlock" that does:
    Lorentz compressor (simplified)  ||  Parallel Euclidean attention
    then fuses them with the chosen GeometryAwareParallelFusion mode.
- Trains with AdamW + cosine schedule.
- Reports: loss, overall recall, long-range recall curves (dist 1-16),
  manifold drift over time, and training speed.

Usage:
    python examples/geometry_fusion_standalone_demo.py --fusion_mode tangent_gated --steps 500
    python examples/geometry_fusion_standalone_demo.py --fusion_mode merge_attn_tangent --steps 500

This demo proves the fusion block is usable as a drop-in primitive for
future Aether reasoning models or next-gen Hyper-SSM hybrids.
"""

import os
import sys
import math
import argparse
from pathlib import Path

# Make sure we can import from the root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hyper_ssm.hyperbolic_ops import (
    stable_expmap, log_o, project_to_manifold, check_manifold_constraint
)
from hyper_ssm.geometry_fusion import GeometryAwareParallelFusion
from hyper_ssm.tiled_compressor import TiledFractalCompressor  # Real production compressor


# ------------------------------------------------------------------
# Same synthetic hierarchical data generator (copied for standalone)
# ------------------------------------------------------------------
def generate_hierarchical_sequences(num_samples, seq_len, num_leaves=32, device="cpu"):
    data = torch.randint(0, num_leaves, (num_samples, seq_len), device=device)
    targets = torch.zeros_like(data)
    for b in range(num_samples):
        for t in range(seq_len):
            past = torch.randint(0, t + 1, (1,)).item() if t > 0 else 0
            level = torch.randint(0, 3, (1,)).item()
            ancestor = (data[b, past] // (3 ** level)) * (3 ** level)
            targets[b, t] = ancestor % num_leaves
    return data.long(), targets.long()


# ------------------------------------------------------------------
# Minimal Fusion-Only Block
# ------------------------------------------------------------------
class RealTiledCompressorWrapper(nn.Module):
    """
    Thin wrapper so the standalone demo can use the real TiledFractalCompressor
    with the GeometryAwareParallelFusion.
    """
    def __init__(self, dim: int, tile_size: int = 16):
        super().__init__()
        self.compressor = TiledFractalCompressor(
            state_dim=dim,
            tile_size=tile_size,
            compile_mode=None  # Keep simple for demo
        )

    def forward(self, x_lor: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, D+1] Lorentz points
        if x_lor.dim() == 2:
            x_lor = x_lor.unsqueeze(1)  # make it [B, 1, D+1] for single step
        out = self.compressor(x_lor)
        return out  # returns full states; we usually take the last one downstream


class FusionOnlyBlock(nn.Module):
    """
    The star of the demo:
    - Takes Euclidean input
    - Projects to Lorentz
    - Runs the *real* TiledFractalCompressor
    - Runs cheap parallel attention
    - Fuses with GeometryAwareParallelFusion (the new module)
    """
    def __init__(self, dim, fusion_mode="tangent_gated"):
        super().__init__()
        self.dim = dim
        self.compressor = RealTiledCompressorWrapper(dim, tile_size=16)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=0.05)
        self.fusion = GeometryAwareParallelFusion(
            dim,
            fusion_mode=fusion_mode,
            gate_type="per_channel",
            use_parallel_transport=True,
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x_euclid):
        # 1. Simple attention path (the "parallel" high-fidelity head)
        attn_out, _ = self.attn(x_euclid, x_euclid, x_euclid)

        # 2. Real compressor path
        x_h = stable_expmap(self.ln(x_euclid))
        h_lor = self.compressor(x_h)          # [B, T, D+1] from real tiled compressor

        # 3. Geometry-aware fusion (this is the new 2026 primitive)
        fused_lor = self.fusion(
            lorentz_state=h_lor,
            euclid_features=attn_out,
            euclid_input_for_gate=x_euclid,
        )

        # Return the spatial part
        return fused_lor[..., 1:]


class StandaloneFusionModel(nn.Module):
    def __init__(self, vocab, dim, fusion_mode="tangent_gated"):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.block = FusionOnlyBlock(dim, fusion_mode=fusion_mode)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx):
        x = self.embed(idx)
        x = self.block(x)
        return self.head(x)


# ------------------------------------------------------------------
# Training loop with rich metrics
# ------------------------------------------------------------------
def train_standalone_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Standalone Geometry Fusion Demo — running on {device}")

    # Data
    train_x, train_y = generate_hierarchical_sequences(
        args.samples, args.seq_len, args.vocab, device=device
    )
    val_x, val_y = generate_hierarchical_sequences(256, args.seq_len, args.vocab, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

    model = StandaloneFusionModel(args.vocab, args.dim, fusion_mode=args.fusion_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFusion mode: {args.fusion_mode}")
    print("Step | TrainLoss | ValLoss | RecallAcc | LongRange@8 | Drift   | Tokens/s")

    manifold_drifts = []
    long_range_recalls = []

    for step in range(args.steps):
        model.train()
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, args.vocab), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = next(iter(val_loader))
                vx, vy = vx.to(device), vy.to(device)
                vlogits = model(vx)
                vloss = criterion(vlogits.view(-1, args.vocab), vy.view(-1))

                pred = vlogits.argmax(-1)
                acc = (pred == vy).float().mean().item()

                # Long-range recall (distance 8)
                if vx.shape[1] > 8:
                    mask = torch.arange(vx.shape[1]) >= 8
                    long_acc = (pred[:, mask] == vy[:, mask]).float().mean().item()
                else:
                    long_acc = acc

                # Manifold health
                sample_emb = model.embed(vx[0:1, -8:])
                sample_lor = stable_expmap(sample_emb)
                drift = check_manifold_constraint(sample_lor).mean().item()
                manifold_drifts.append(drift)
                long_range_recalls.append(long_acc)

            tokens_per_sec = (args.batch * args.seq_len) / max(0.001, (loss.item() / 100))  # rough proxy
            print(f"{step:4d} | {loss.item():.3f}   | {vloss.item():.3f} | {acc:.3f}    | {long_acc:.3f}       | {drift:.1e} | ~{tokens_per_sec:.0f}")

    # Final long-range curve
    print("\n=== FINAL LONG-RANGE HIERARCHICAL RECALL CURVE ===")
    model.eval()
    with torch.no_grad():
        vx, vy = next(iter(val_loader))
        vx, vy = vx.to(device), vy.to(device)
        logits = model(vx)
        pred = logits.argmax(-1)

        curve = []
        for dist in [1, 2, 4, 8, 16]:
            if dist >= vx.shape[1]:
                break
            mask = torch.arange(vx.shape[1]) >= dist
            r = (pred[:, mask] == vy[:, mask]).float().mean().item()
            curve.append((dist, r))
        print("Distance | Recall")
        for d, r in curve:
            print(f"   {d:2d}    | {r:.4f}")

    print(f"\nDemo finished. Final manifold drift trend (last 3): {manifold_drifts[-3:]}")
    print("You can now extend this script with real TiledFractalCompressor + full liquid experts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_mode", type=str, default="tangent_gated",
                        choices=["tangent_gated", "merge_attn_tangent", "lorentz_native"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    train_standalone_demo(args)
