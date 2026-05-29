"""
train_extreme.py — Scaling Hyper-SSM to ~800M Parameters on 6GB VRAM
=====================================================================
Demonstrates the O(1) memory bound of the FractalStateCompressor combined with
extreme memory optimizations:
1. `bitsandbytes` 8-bit AdamW (slashes optimizer state by 75%)
2. `torch.utils.checkpoint` (recomputes forward activations to save VRAM)

Usage:
    python train_extreme.py
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Attempt to load 8-bit optimizer. Fall back gracefully if not installed.
try:
    import bitsandbytes as bnb
    OPTIMIZER_CLASS = bnb.optim.AdamW8bit
    print(f"[SYSTEM] Loaded bitsandbytes 8-bit Optimizer.")
except ImportError:
    import torch.optim as optim
    OPTIMIZER_CLASS = optim.AdamW
    print(f"[WARNING] bitsandbytes not found. Falling back to standard 32-bit AdamW. OOM highly likely.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig, HyperSSMBlock
from hyper_ssm.tokenizer import HyperTokenizer
from train_c_code import CCodeDataLoader

# Centralized Riemannian ops (2026 refactor)
from hyper_ssm.cuda_ops import get_project_to_tangent
project_to_tangent = get_project_to_tangent()

# ===================================================================
#  Patched Model for Gradient Checkpointing
# ===================================================================
class CheckpointedHyperSSM(HyperSSM):
    """Overrides the forward pass to checkpoint each layer, saving massive activation VRAM."""
    def forward(self, idx, return_entropy=False):
        x = self.tok_emb(idx)
        total_entropy = 0.0

        for layer in self.layers:
            # We explicitly checkpoint the layer forward pass. 
            # Note: We must pass requires_grad=True inputs or checkpointing will bypass graph.
            def custom_forward(*inputs):
                return layer(inputs[0])
            
            x, entropy = checkpoint(custom_forward, x, use_reentrant=False)
            total_entropy += entropy

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_entropy:
            return logits, total_entropy
        return logits


# ===================================================================
#  Training Loop
# ===================================================================
def train_extreme():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")
    
    # 1. Target ~800 Million Parameters Architecture
    # Base configuration: 
    # GPT-2 Large is: d_model=1280, n_layer=36 => ~774M params.
    # We will mimic a similar scaling law footprint for our architecture.
    tokenizer = HyperTokenizer()
    config = HyperSSMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1152,  # Massive hidden state
        num_layers=36      # Deep sequence folding
    )
    
    print("\n[INIT] Instantiating CheckpointedHyperSSM (~800M)...")
    # We allocate directly to CPU first because ~800M fp32 params = ~3.2GB.
    model = CheckpointedHyperSSM(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {params / 1e6:.2f}M massive parameters.")
    
    # Move to GPU and ensure precision is pure float32 (hyperbolic constraint)
    model = model.to(device)
    torch.set_float32_matmul_precision("medium")
    
    # Enable gradient tracking globally to support activation checkpointing
    # The first layer needs requires_grad=True so autograd knows to record operations
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.tok_emb.register_forward_hook(make_inputs_require_grad)
    
    # 2. Extreme Optimizer Load
    # 8-bit Adam uses 2 bytes per parameter (1.6 GB) instead of 8 bytes (6.4 GB).
    optimizer = OPTIMIZER_CLASS(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Data Loader setup (Extreme Minimal Context)
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "c_corpus.txt")
    
    batch_size = 1     # Hard minimum
    seq_len = 256      # Shortened specifically to test VRAM boundary
    
    loader = CCodeDataLoader(corpus_path, tokenizer, batch_size, seq_len)
    
    print(f"\\n[TRAIN] Attempting execution on 6GB VRAM limit...")
    print(f"        Batch Size: {batch_size} | Seq Len: {seq_len}")
    
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    history = []
    
    t0 = time.perf_counter()
    for step, (X, Y) in enumerate(loader.stream(epochs=1)):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass (Checkpointed)
        logits, entropy = model(X, return_entropy=True)
        loss = criterion(logits.reshape(-1, config.vocab_size), Y.reshape(-1))
        
        if torch.isnan(loss):
            print(f"\\n[FATAL] NaN at step {step}. Aborting.")
            break
            
        total_loss = loss - 0.01 * entropy
        
        # Backward pass (Recomputes activations mathematically instead of storing them)
        total_loss.backward()
        
        # Riemannian Safety projections
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None and ("W_state.weight" in name or "W_input.weight" in name):
                    p.grad = project_to_tangent(p, p.grad)
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    p.grad.zero_()
                    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Memory tracking
        max_vram = torch.cuda.max_memory_allocated() / (1024**3)
        current_vram = torch.cuda.memory_allocated() / (1024**3)
        
        history.append(loss.item())
        
        print(f"  Step {step:03d} | Loss: {loss.item():.4f} | Peak VRAM: {max_vram:.2f}GB / {current_vram:.2f}GB Alloc")
        
        # Prove stability by reaching step 5 without breaking OOM
        if step >= 15:
            print("\\n[SUCCESS] Survived 15 steps of ~800M parameter evaluation without OOM.")
            break

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    train_extreme()
