# Hyper-SSM Ultimate: Positioning for 2026 Research

## Title Suggestion
**Hyper-SSM Ultimate: Hybrid Lorentzian State-Space Models with Liquid Experts and Selective Recall Attention**

## Abstract (Draft)

We present Hyper-SSM Ultimate, a hybrid architecture that combines exact Lorentzian manifold recurrence for constant-memory state compression with hypernetwork-synthesized dynamic experts and sparse high-fidelity attention layers. Building on advances in hyperbolic geometry for sequence modeling (HiM, 2025) and production hybrid SSM-Transformer designs (NVIDIA Nemotron 3, 2026), our model achieves strong hierarchical reasoning while maintaining practical O(1) persistent state. We introduce tiled variants of the fractal compressor inspired by NVIDIA's cuTile programming model and provide a clear path for native implementation in the newly released cuda-oxide Rust-to-CUDA compiler. Experiments on embedded C firmware and long-context tasks demonstrate competitive performance with significantly improved memory scaling compared to pure Transformer and Mamba baselines.

## Key Differentiators (2026 Landscape)

1. **True Geometric O(1) State**  
   Unlike most hybrid Mamba-Transformer models that still rely on growing or fixed-size hidden states with linear attention approximations, we use an explicit Lorentzian recursive compressor with learnable curvature.

2. **Dynamic Liquid Computation**  
   Our hypernetwork-synthesized experts (liquid weights) remain one of the few serious attempts at context-dependent parameter generation at inference time — going beyond standard MoE routing.

3. **Principled Hyperbolic Losses**  
   Direct adoption and extension of centripetal + clustering losses from the HiM line of work, applied to a full generative language model rather than just embedding tasks.

4. **Modern Kernel Roadmap**  
   Explicit support for both NVIDIA's cuTile (productivity + performance) and cuda-oxide (Rust safety + performance) — positioning the work at the cutting edge of 2026 GPU programming models.

## Related Work Placement

- **Vs. HiM / HMamba / SHMamba (2025)**: We move from classification/embedding tasks to full autoregressive language modeling with dynamic experts.
- **Vs. Nemotron-H / Nemotron 3 (2025-2026)**: We replace the Mamba-2 backbone with a geometrically richer Lorentzian compressor while keeping the hybrid philosophy.
- **Vs. Pure Hyperbolic Transformers (Hypformer, HELM, etc.)**: We avoid quadratic attention for the majority of layers.

## Contributions

- Hybrid Lorentzian + attention architecture with strong empirical motivation.
- Tiled compressor design aligned with emerging tile-based GPU programming models.
- First public integration path from a geometric SSM to NVIDIA's cuda-oxide (May 2026).
- Open implementation with modern engineering practices.

This work sits at a timely intersection of geometric deep learning, efficient sequence modeling, and the next generation of GPU programming abstractions.
