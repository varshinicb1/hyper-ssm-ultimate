# Hyper-SSM Ultimate (2026)

**The best-integrated version of Hyper-SSM: Lorentzian Fractal State Compression + Liquid Experts + Hybrid Recall Attention**

This repository contains the 2026 "Ultimate" realization of the original Hyper-SSM vision, fully updated with the latest research and production insights as of May 2026.

## Core Idea (Still Unique and Powerful)

Replace the quadratic KV-cache and static parameters of Transformers with:

1. **Lorentzian Fractal State Compressor** — Recursive folding of sequence history into fixed-size hyperbolic state (O(1) memory).
2. **Liquid Mixture-of-Experts** — A small hypernetwork dynamically synthesizes transient expert weights on every forward pass.
3. **Hybrid Recall Layers** (new in Ultimate) — Occasional high-quality attention layers for precise retrieval (following NVIDIA Nemotron 3 philosophy).

## Major 2026 Upgrades

- **Hybrid Architecture** — Geometric compressor for most layers + selective attention for recall (best of both worlds).
- **Hyperbolic Structural Losses** — Centripetal + clustering losses (inspired by HiM, arXiv:2505.18973).
- **Modern Tooling**:
  - Centralized CUDA Riemannian ops (`hyper_ssm/cuda_ops.py`)
  - `torch.compile`-friendly compressor
  - cuTile-inspired tiled compressor prototype
  - Explicit path for **NVIDIA cuda-oxide** (Rust → PTX, released May 2026)
- Clean, reproducible, well-documented codebase.

## Key Files

- `HYPER_SSM_2026_ULTIMATE.md` — The full integrated vision and roadmap
- `hyper_ssm/model.py` — `HyperSSM` with `use_hybrid=True` support + `HybridHyperSSMBlock`
- `hyper_ssm/hyperbolic_loss.py` — HiM-style losses
- `training/train_hybrid_ultimate.py` — Recommended training script
- `rust_kernels/` — cuda-oxide integration skeleton
- `hyper_ssm/tiled_compressor.py` — cuTile-inspired optimization direction

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but recommended) Build the CUDA extension
python compile_cuda.py build_ext --inplace

# 3. Try the flagship hybrid training script (demo)
python training/train_hybrid_ultimate.py --epochs 1 --batch 2 --seq_len 256
```

## Positioning vs 2026 Landscape

- Vs **HiM** (Hierarchical Mamba + Lorentz): We add powerful dynamic liquid experts + hybrid attention.
- Vs **NVIDIA Nemotron 3 / hybrid Mamba-Transformer**: We bring true geometric O(1) hyperbolic state + on-the-fly parameter synthesis.
- Vs pure geometric models: We adopted the hard lesson that selective attention recall layers are necessary for frontier quality.

## Future Directions (High Leverage)

1. Full cuTile / native CUDA implementation of the tiled compressor.
2. Port the core recurrence to **cuda-oxide** (Rust) for maximum performance + safety.
3. Large-scale training runs with proper long-context evaluations.
4. Multimodal extensions using the existing vision/audio topology modules.

This project is now one of the most coherent and ambitious geometric + dynamic-compute architectures available.

See `HYPER_SSM_2026_ULTIMATE.md` for the complete technical vision.
