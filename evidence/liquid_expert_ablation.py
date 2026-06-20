"""
Evidence Script: Liquid Experts vs Standard FFN Ablation

Template for comparing:
- Full model with DynamicLiquidLayer (our liquid experts)
- Same architecture but with standard FFN in place of the liquid experts

For real evidence this should be run at 100M+ scale with proper training budget.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.liquid_weights import DynamicLiquidLayer

class StandardFFN(nn.Module):
    """Standard feed-forward replacement for ablation (no dynamic synthesis)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()

    def forward(self, x, context_state=None):
        # context_state is ignored — this is the key difference from liquid experts
        return self.fc2(self.act(self.fc1(x)))

def create_ablated_model(use_liquid: bool = True, hidden_size: int = 128, num_layers: int = 6):
    config = HyperSSMConfig(vocab_size=1000, hidden_size=hidden_size, num_layers=num_layers)
    model = HyperSSM(config, use_hybrid=False, use_tiled_compressor=True)

    if not use_liquid:
        for layer in model.layers:
            if hasattr(layer, "liquid_mlp"):
                layer.liquid_mlp = StandardFFN(hidden_size)

    return model

if __name__ == "__main__":
    print("Creating ablation models (Liquid Experts vs Standard FFN)...")
    model_liquid = create_ablated_model(use_liquid=True)
    model_ffn = create_ablated_model(use_liquid=False)

    print(f"With Liquid Experts: {sum(p.numel() for p in model_liquid.parameters())/1e6:.2f}M params")
    print(f"With Standard FFN:  {sum(p.numel() for p in model_ffn.parameters())/1e6:.2f}M params")

    print("\nReady for matched training runs to measure the value of dynamic liquid experts.")
    print("Use training/train_hybrid_ultimate.py with manual layer replacement for real ablations.")
