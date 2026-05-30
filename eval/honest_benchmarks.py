"""
Honest Benchmarking Script for Hyper-SSM

Goal: Measure actual capabilities without hype.
This script prioritizes:
- Proper baselines
- Clear success/failure criteria
- Reproducible methodology
- Explicit caveats

Current Status: First real experiment being built.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
import json
from datetime import datetime
import random
import numpy as np

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyper_ssm import HyperSSM, HyperSSMConfig, create_hyperbolic_loss
from hyper_ssm.hyperbolic_ops import stable_expmap, project_to_manifold

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# =============================================================================
# SMALLEST MEANINGFUL EXPERIMENT: Hierarchical Recall with Geometric Loss
# =============================================================================
#
# Core Hypothesis Being Tested:
# "The geometrically correct HyperbolicLoss (applied to real Lorentz states
#  in tangent space) improves long-range hierarchical recall compared to:
#   (a) No auxiliary loss
#   (b) A standard Euclidean auxiliary loss"
#
# Why this is the smallest meaningful experiment:
# - Directly tests whether making the loss "actually geometric" (the fix done in June 2026) matters.
# - Uses a controllable synthetic task that matches one of the project's claimed strengths.
# - Small models + short training = runnable on CPU in reasonable time.
# - Has clear, falsifiable success criteria.
#
# Task:
# Sequences where each token has a hierarchical label (tree structure).
# Model must recall the correct ancestor label from a distant past position.
# We measure accuracy as a function of distance in the sequence.
# =============================================================================

def generate_hierarchical_data(num_samples: int, seq_len: int, num_labels: int = 32, depth: int = 3):
    """
    Generate synthetic data with hierarchical structure.
    Each position has a label. Ancestors exist at different levels.
    """
    data = torch.randint(0, num_labels, (num_samples, seq_len))
    targets = torch.zeros_like(data)

    for b in range(num_samples):
        for t in range(seq_len):
            if t == 0:
                targets[b, t] = data[b, t]
                continue
            # Pick a random past position and a random ancestor level
            past = random.randint(0, t)
            level = random.randint(0, depth - 1)
            # Crude hierarchy: higher bits represent higher ancestors
            divisor = max(1, num_labels // (2 ** (level + 1)))
            ancestor = (data[b, past] // divisor) * divisor
            targets[b, t] = ancestor % num_labels

    return data.long(), targets.long()


class EuclideanAuxLoss(nn.Module):
    """Simple Euclidean contrastive-style auxiliary loss for baseline comparison."""
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]
        # Very simple: encourage some structure in Euclidean space
        B, T, D = hidden_states.shape
        if T < 4:
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Sample some pairs
        anchors = hidden_states[:, :-2, :].reshape(-1, D)
        positives = hidden_states[:, 1:-1, :].reshape(-1, D)
        negatives = hidden_states[:, 2:, :].reshape(-1, D)
        
        pos_dist = (anchors - positives).norm(dim=-1).mean()
        neg_dist = (anchors - negatives).norm(dim=-1).mean()
        loss = torch.relu(pos_dist - neg_dist + 0.5).mean()
        return self.weight * loss


def run_hierarchical_recall_experiment(
    model_dim: int = 64,
    num_layers: int = 4,
    seq_len: int = 128,
    num_train_steps: int = 300,
    batch_size: int = 8,
    device: str = "cpu",
    output_dir: Path = Path("eval_results/hierarchical_recall")
):
    """
    The actual smallest meaningful experiment.
    Trains tiny models under three conditions and measures hierarchical recall at distance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir()

    print(f"\n=== Running Smallest Meaningful Experiment ===")
    print(f"Output: {run_dir}")
    print(f"Model: {num_layers} layers, dim={model_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Training steps: {num_train_steps}")

    conditions = {
        "no_aux_loss": None,
        "euclidean_aux": EuclideanAuxLoss(weight=0.01),
        "hyperbolic_loss": create_hyperbolic_loss(centripetal_weight=0.01, clustering_weight=0.01, tangent_space=True),
    }

    all_results = {}

    for cond_name, aux_loss_fn in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")

        config = HyperSSMConfig(vocab_size=32, hidden_size=model_dim, num_layers=num_layers)
        model = HyperSSM(config, use_hybrid=True, use_tiled_compressor=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training
        model.train()
        for step in range(num_train_steps):
            x, y = generate_hierarchical_data(batch_size, seq_len, num_labels=32)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            lm_loss = F.cross_entropy(logits.view(-1, 32), y.view(-1))

            total_loss = lm_loss
            if aux_loss_fn is not None:
                # Get Lorentz states for hyperbolic loss, or hidden for Euclidean
                if cond_name == "hyperbolic_loss":
                    with torch.no_grad():
                        lorentz_states = model.get_lorentz_representations(x, final_only=True)["lorentz_states"]
                    loss_dict = aux_loss_fn(lorentz_states)
                    aux = torch.tensor(loss_dict.get("total", 0.0), device=device, dtype=lm_loss.dtype)
                else:
                    # For Euclidean baseline, use last layer hidden (approximate)
                    hidden = model.tok_emb(x)  # crude
                    aux = aux_loss_fn(hidden)
                total_loss = lm_loss + aux

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  Step {step:3d} | LM Loss: {lm_loss.item():.3f}")

        # Evaluation: Hierarchical Recall at Distance
        model.eval()
        distances = [8, 16, 32, 64]
        recall_results = {}

        with torch.no_grad():
            for dist in distances:
                correct = 0
                total = 0
                for _ in range(20):  # evaluation batches
                    x, y = generate_hierarchical_data(4, seq_len, num_labels=32)
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)

                    # Evaluate only positions where we can look back `dist` steps
                    for b in range(x.size(0)):
                        for t in range(dist, seq_len):
                            if preds[b, t] == y[b, t]:
                                correct += 1
                            total += 1

                acc = correct / max(total, 1)
                recall_results[f"dist_{dist}"] = round(acc * 100, 2)
                print(f"  Recall@dist={dist}: {acc*100:.1f}%")

        all_results[cond_name] = {
            "recall": recall_results,
            "final_lm_loss": round(lm_loss.item(), 4)
        }

    # Save results
    summary = {
        "timestamp": timestamp,
        "experiment": "hierarchical_recall_with_geometric_loss",
        "hypothesis": "Geometrically correct HyperbolicLoss improves long-range hierarchical recall vs Euclidean aux loss and no aux loss",
        "model_config": {"dim": model_dim, "layers": num_layers, "seq_len": seq_len},
        "results": all_results,
        "caveats": [
            "Very small model and short training.",
            "Synthetic task only.",
            "This is the minimal viable test of whether the geometric loss fix matters.",
            "Larger scale experiments are still required."
        ]
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {run_dir / 'results.json'}")
    print("This is a real (if tiny) experiment. Treat results as directional only.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Honest Hyper-SSM Benchmarking")
    parser.add_argument("--experiment", type=str, default="hierarchical_recall",
                        choices=["hierarchical_recall", "memory_scaling"],
                        help="Which experiment to run")
    parser.add_argument("--steps", type=int, default=300, help="Training steps for the experiment")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.experiment == "hierarchical_recall":
        run_hierarchical_recall_experiment(
            model_dim=args.dim,
            num_layers=args.layers,
            seq_len=args.seq_len,
            num_train_steps=args.steps,
            output_dir=output_dir / "hierarchical_recall"
        )
    else:
        print("Other experiments not implemented yet.")


# =============================================================================
# BRIDGE: Real Utility Evaluation for Self-Improvement Engine
# =============================================================================

def evaluate_system_config(
    config: Dict[str, Any],
    model_dim: int = 48,
    num_layers: int = 3,
    seq_len: int = 64,
    num_train_steps: int = 200,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    This is the key bridge function for mathematical self-improvement.

    The SelfImprovementEngine in Aether calls this with a proposed configuration
    (loss weights, fusion mode, etc.). This function trains a model under that
    configuration on the hierarchical recall task and returns the utility
    (especially recall at longer distances).

    This makes the self-improvement engine optimize against *real* benchmark
    results instead of heuristics.
    """
    print(f"[evaluate_system_config] Evaluating config: {config}")

    hyp_weight = float(config.get("hyperbolic_loss_weight", 0.01))
    cent_weight = float(config.get("centripetal_weight", 0.003))
    clust_weight = float(config.get("clustering_weight", 0.003))
    lr = float(config.get("learning_rate", 1e-3))

    custom_hyp_loss = create_hyperbolic_loss(
        centripetal_weight=cent_weight,
        clustering_weight=clust_weight,
        tangent_space=True,
    )

    config_obj = HyperSSMConfig(vocab_size=32, hidden_size=model_dim, num_layers=num_layers)
    model = HyperSSM(config_obj, use_hybrid=True, use_tiled_compressor=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(num_train_steps):
        x, y = generate_hierarchical_data(batch_size, seq_len, num_labels=32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        lm_loss = F.cross_entropy(logits.view(-1, 32), y.view(-1))

        with torch.no_grad():
            lorentz_states = model.get_lorentz_representations(x, final_only=True)["lorentz_states"]

        loss_dict = custom_hyp_loss(lorentz_states)
        hyp_component = torch.tensor(loss_dict.get("total", 0.0), device=device, dtype=lm_loss.dtype)

        total_loss = lm_loss + hyp_weight * hyp_component

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  [Self-Imp] Step {step} | LM={lm_loss.item():.3f}")

    # Evaluation
    model.eval()
    distances = [8, 16, 32, 64]
    recall = {}

    with torch.no_grad():
        for dist in distances:
            correct = 0
            total = 0
            for _ in range(12):
                x, y = generate_hierarchical_data(4, seq_len, num_labels=32)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)

                for b in range(x.size(0)):
                    for t in range(dist, seq_len):
                        if preds[b, t] == y[b, t]:
                            correct += 1
                        total += 1

            acc = correct / max(1, total)
            recall[f"recall_dist_{dist}"] = round(acc * 100, 2)

    utility = recall.get("recall_dist_32", 0.0)   # Primary optimization target

    result = {
        "utility": utility,
        "recall": recall,
        "final_lm_loss": round(lm_loss.item(), 4),
        "config_used": config,
    }

    print(f"[evaluate_system_config] → Utility (recall@32) = {utility:.2f}")
    return result


if __name__ == "__main__":
    main()