# Hyper-SSM Ultimate

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Research_Artifact-6B46C1)](https://github.com/varshinicb1/hyper-ssm-ultimate)

[![GitHub stars](https://img.shields.io/github/stars/varshinicb1/hyper-ssm-ultimate?style=social)](https://github.com/varshinicb1/hyper-ssm-ultimate/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/varshinicb1/hyper-ssm-ultimate)](https://github.com/varshinicb1/hyper-ssm-ultimate/commits/main)
[![Geometric Deep Learning](https://img.shields.io/badge/Geometric_DL-2026-purple)](https://github.com/varshinicb1/hyper-ssm-ultimate)

**Lorentzian Fractal State-Space Models with Liquid Experts and Hybrid Recall Attention**

An advanced research artifact exploring geometric alternatives to the Transformer KV-cache, combining exact Lorentzian manifold recurrence for constant-memory state with hypernetwork-synthesized dynamic experts.

**Repository:** [github.com/varshinicb1/hyper-ssm-ultimate](https://github.com/varshinicb1/hyper-ssm-ultimate)

---

## Current Status (Honest)

This repository contains a **significantly hardened and productionized research prototype** developed in May 2026.

**What has been built:**
- A complete, modular Python implementation of the core architecture.
- A high-performance `TiledFractalCompressor` with heavy vectorization, `torch.compile` support, and numerical stability across precisions.
- A real (not sketched) Rust implementation with PyO3 interop for the compressor logic.
- A modern, robust training infrastructure capable of long experiments.
- Clean hybrid architecture (geometric state + selective attention recall layers).

**What this is not (yet):**
- A fully trained large-scale model with published benchmark numbers against Llama-3 / Qwen / Nemotron equivalents.
- A drop-in production inference library.
- A finished research paper with rigorous ablations and scaling curves.

The goal of this repository is to provide a high-quality, honest, and extensible foundation for continued research in this direction.

---

## What Exists in the Repository Today

### Core Architecture (`hyper_ssm/`)
- `HybridHyperSSM` and `HybridHyperSSMBlock`: Full hybrid model supporting both pure geometric and geometric + selective attention modes.
- `TiledFractalCompressor`: The flagship production-grade compressor (cuTile-inspired tiling, aggressive vectorization inside tiles, `torch.compile` with robust fallbacks, `get_final_state` + `update_state` for efficient generation, manifold projection/repair, performance counters, and autotuning).
- `DynamicLiquidLayer` + `HyperWeightSynthesizer`: The original liquid MoE mechanism with spectral normalization and entropy regularization.
- `HyperbolicLoss`: HiM-style centripetal + clustering losses for better hierarchical structure learning.
- `SelectiveAttentionRecall`: Lightweight attention layers for high-fidelity recall.
- Centralized Riemannian operations (`cuda_ops.py`) with clean CUDA fallback.

### Rust / cuda-oxide Kernels (`rust_kernels/`)
- A complete, buildable Rust crate with:
  - Exact-parity Lorentzian operations (product, normalization, gated Einstein midpoint).
  - Full single-tile and tiled compressor implementations (with shared-memory patterns and explicit barrier points documented for future GPU porting).
  - PyO3 + numpy zero-copy bindings (buildable today via `maturin`).
  - Rayon parallel batch processing for strong CPU performance.
  - Clear GPU porting recipe targeting clusters + TMA + WGMMA/tcgen05.
- The Python `TiledFractalCompressor` can optionally delegate to these Rust kernels.

### Training & Infrastructure
- `training/train_hybrid_ultimate.py`: A genuinely production-grade training script featuring:
  - Automatic mixed precision (bf16/fp16/fp32) with GradScaler
  - Cosine learning rate with warmup
  - Atomic, crash-safe checkpointing (full RNG state, optimizer, scheduler, scaler)
  - Rich JSONL + manifest logging with system info and performance counters
  - Gradient accumulation
  - Startup autotuning of compilation
  - First-class `--use_tiled` and `--use_rust_accel` flags
  - Robust resume support

### Examples, Benchmarks & Documentation
- `examples/production_usage.py`: Clean demonstration of model usage, generation APIs, and Rust acceleration.
- `benchmark_compressor.py`: Extensive benchmark including speed, numerical stability, manifold violation checks, and Python vs Rust parity tests.
- `HYPER_SSM_2026_ULTIMATE.md`: Detailed technical vision and roadmap.
- `docs/PAPER_POSITIONING_2026.md`: Positioning against HiM, Nemotron 3, pure hyperbolic models, etc.

### Data & Legacy
- Real embedded C firmware corpus (`data/c_corpus.txt`, ~10.5M tokens from FreeRTOS + ESP-IDF + NXP).
- Preserved original research notes and legacy experiments in `legacy/`.

---

## Vision

Hyper-SSM aims to explore whether **geometric state compression + dynamic parameter synthesis** can offer a compelling alternative (or complement) to the dominant Transformer + KV-cache paradigm.

**Core hypothesis:**
A combination of Lorentzian manifold recurrence (for O(1) persistent state with exponential capacity) and hypernetwork-generated transient experts (for context-dependent computation) can deliver strong performance with dramatically better memory scaling than attention, especially when hybridized with a small number of high-fidelity recall layers.

**Long-term direction:**
- Mature, high-performance kernels via NVIDIA cuda-oxide (Rust) and/or cuTile.
- Rigorous scaling studies and long-context evaluations.
- Multimodal extensions (vision/audio topologies already exist in the codebase).
- Open research artifact suitable for collaboration and follow-up work.

This work sits at the intersection of hyperbolic geometry in deep learning (HiM and related 2025 work), production hybrid architectures (NVIDIA Nemotron 3 family, 2026), and the emerging next generation of GPU programming models (cuTile + cuda-oxide).

---

## Getting Started

### Installation

```bash
git clone https://github.com/varshinicb1/hyper-ssm-ultimate.git
cd hyper-ssm-ultimate

pip install -r requirements.txt

# Optional: Build the legacy CUDA extension (Riemannian ops)
python compile_cuda.py build_ext --inplace
```

### Quick Smoke Test (Hybrid + Tiled)

```bash
python training/train_hybrid_ultimate.py \
  --epochs 1 \
  --batch 2 \
  --seq_len 256 \
  --use_tiled \
  --max_steps 50
```

### Production-Style Training Example

```bash
python training/train_hybrid_ultimate.py \
  --use_tiled \
  --precision auto \
  --max_steps 50000 \
  --batch 8 \
  --seq_len 1024 \
  --grad_accum 4 \
  --autotune_compile \
  --log_interval 100
```

See `examples/production_usage.py` for generation and Rust acceleration examples.

---

## Limitations (Transparency)

- Most experiments in this repository have been run at modest scale (tens to low hundreds of millions of parameters) due to hardware constraints.
- The Rust kernels currently provide a high-quality CPU reference + clear GPU porting path; full native cuda-oxide GPU kernels are not yet complete.
- Strong long-context needle-in-a-haystack and scaling law results are still future work.
- Some large checkpoint files and experimental outputs have been excluded via `.gitignore`.

---

## Roadmap (High-Leverage Next Steps)

1. **Kernel maturation** — Complete high-performance single-tile and tiled kernels in cuda-oxide (or Triton/cuTile).
2. **Rigorous evaluation** — Long-context benchmarks, scaling curves, and direct comparisons against strong Mamba-2 and Nemotron-style hybrids.
3. **Multimodal** — Leverage the existing vision and audio topology modules for non-text experiments.
4. **Open research** — Clean up for potential open-sourcing or collaboration.

---

## Acknowledgments & Positioning

This project builds upon ideas from:
- Hyperbolic geometry in deep learning (Nickel et al., HiM 2025, HELM, Hypformer, etc.)
- State-space models (S4, Mamba/Mamba-2)
- Production hybrid architectures (NVIDIA Nemotron-H / Nemotron 3, 2025–2026)
- Dynamic and liquid computation ideas

The engineering push in this repository (May 2026) focused on turning promising geometric ideas into something that can actually be trained seriously and extended.

---

**If you are a researcher or engineer interested in geometric alternatives to attention or hybrid SSM-Transformer designs, this repository is intended as a high-quality starting point rather than a finished product.**

Contributions, discussions, and collaborations are welcome.