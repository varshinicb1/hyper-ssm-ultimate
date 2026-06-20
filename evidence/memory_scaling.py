"""
Evidence Script: Memory Scaling Curves for Hyper-SSM Compressor

Compares:
- TiledFractalCompressor (our production version)
- Naive per-token compressor (original style)
- Simple causal attention simulation (upper bound for standard Transformer KV cache memory)
- Simple recurrent block (proxy for Mamba-2 style state)

This script measures peak memory (RSS on CPU, or CUDA allocated if GPU is available)
as sequence length increases.

NOTE: On this machine we are likely CPU-only. CPU RSS is a rough proxy.
On real GPU hardware, replace the memory measurement with torch.cuda.max_memory_allocated().

Run with increasing sequence lengths to generate the curve.
"""

import torch
import time
import psutil
import os
import sys

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyper_ssm.tiled_compressor import TiledFractalCompressor
from hyper_ssm.hyperbolic_ops import FractalStateCompressor, lorentz_normalize

def get_memory_mb():
    """Return current peak memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    else:
        # For CPU we can't easily reset RSS, so we just measure delta
        pass

def measure_peak_memory(fn, device):
    """Run fn and return peak memory used during execution."""
    reset_memory()
    mem_before = get_memory_mb()
    fn()
    mem_after = get_memory_mb()
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        return max(0, mem_after - mem_before)

def create_attention_memory_proxy(seq_len, hidden_size, batch=1, num_layers=12):
    """
    Rough proxy for standard Transformer KV cache memory.
    For causal attention, KV cache is roughly 2 * batch * seq_len * hidden * num_layers * 2 bytes (fp16) or 4 (fp32).
    We simulate by allocating the equivalent tensors.
    """
    dtype = torch.float32
    kv_size = 2 * batch * seq_len * hidden_size * num_layers  # K + V
    kv = torch.empty(kv_size, dtype=dtype, device='cpu')  # Force CPU for fair comparison
    return kv  # Keep reference so it isn't GC'd during measurement

def theoretical_attention_kv_mb(batch, seq_len, hidden_size, num_layers=12, bytes_per_param=4):
    """Theoretical memory for standard Transformer KV cache (fp32)."""
    # 2 (K+V) * batch * seq * hidden * layers * bytes
    return (2 * batch * seq_len * hidden_size * num_layers * bytes_per_param) / (1024 ** 2)

def theoretical_mamba_state_mb(batch, hidden_size, state_dim=16, num_layers=12, bytes_per_param=4):
    """Rough theoretical for Mamba-style selective SSM state (constant in seq_len)."""
    # Very approximate: hidden * state_dim * layers * 2 (for conv/state) or similar
    return (batch * hidden_size * state_dim * num_layers * bytes_per_param) / (1024 ** 2)

def run_scaling_experiment(state_dim=256, batch=2, seq_lengths=None, iters=1):
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Batch: {batch}, State dim: {state_dim}")
    print("-" * 80)
    header = f"{'Seq Len':>10} | {'Tiled (MB)':>12} | {'Naive (MB)':>12} | {'Attn Theo (MB)':>15} | {'Mamba Theo (MB)':>17}"
    print(header)
    print("-" * len(header))

    results = []

    for sl in seq_lengths:
        x = torch.randn(batch, sl, state_dim + 1, device=device)
        x = lorentz_normalize(x)

        # Tiled
        tiled = TiledFractalCompressor(state_dim, tile_size=64, compile_mode=None).to(device).eval()
        def run_tiled():
            _ = tiled(x)
        mem_tiled = measure_peak_memory(run_tiled, device)
        del tiled

        # Naive
        naive = FractalStateCompressor(state_dim, compile_mode=None).to(device).eval()
        def run_naive():
            _ = naive(x)
        mem_naive = measure_peak_memory(run_naive, device)
        del naive

        # Theoretical baselines
        mem_attn_theo = theoretical_attention_kv_mb(batch, sl, state_dim)
        mem_mamba_theo = theoretical_mamba_state_mb(batch, state_dim)

        print(f"{sl:>10} | {mem_tiled:>12.1f} | {mem_naive:>12.1f} | {mem_attn_theo:>15.1f} | {mem_mamba_theo:>17.1f}")

        results.append({
            "seq_len": sl,
            "tiled_mb": round(mem_tiled, 2),
            "naive_mb": round(mem_naive, 2),
            "attn_theoretical_mb": round(mem_attn_theo, 2),
            "mamba_theoretical_mb": round(mem_mamba_theo, 2)
        })

        # Cleanup
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("-" * len(header))
    print("\nResults JSON (copy for plotting):")
    import json
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    run_scaling_experiment()
