# Title (Draft)
Hyper-SSM: Lorentzian State-Space Models with Liquid Mixture-of-Experts

> **Status Note (June 2026):** This is an earlier draft paper. The implementation has since evolved significantly into "Hyper-SSM Ultimate 2026" with tiled compressors, hybrid attention, GeometryAwareParallelFusion, geometrically correct hyperbolic losses on actual Lorentz states, and a hardened production training stack. The most up-to-date description lives in the repository README and `HYPER_SSM_2026_ULTIMATE.md`.

## Abstract
We propose Hyper-SSM, a hyperbolic state-space architecture combining exact Lorentzian manifold operations with hypernetwork-synthesized mixture-of-experts layers. The model maintains O(1) memory state while enabling dynamic computational expansion through transient parameter instantiation. We demonstrate stable training up to 40M parameters under bfloat16 precision and evaluate scaling behavior, long-context retention, and multimodal compatibility.

## 1. Introduction
Modern Large Language Models (LLMs) depend largely on the Transformer architecture. However, the Multi-Head Self-Attention mechanism exhibits a quadratic memory scaling constraint known as the KV-Cache. To address these geometric limitations, Hyper-SSM leverages Hyperbolic geometry for hierarchical compression, effectively packing sequence states into fixed-dimensional vectors, and introduces Liquid Hypernetworks for dynamic parameter expansion without persistent storage overhead.

## 2. Mathematical Foundation

### 2.1 Lorentzian Hyperboloid Model
The architecture maps inputs to a Lorentzian manifold where the scalar product satisfies:
$\langle x, x \rangle_L = -1$
This geometry expands exponentially with its radius, providing theoretically unbounded volumetric capacity to store complex hierarchical syntactic structures compared to Euclidean approximations.

### 2.2 Einstein Midpoint
Sequence states are blended not via linear Euclidean additions, but exclusively using the exact midpoint in Minkowski space. This prevents sequential history from drifting off the manifold during recurrent iterations.

### 2.3 Riemannian Gradient Projection
$g_T = g + \langle x, g \rangle_L x$
To successfully learn topologies, the optimizer maps standard Euclidean gradients into the tangent space of the activation state.

## 3. Architecture

### 3.1 Hyperbolic State Update
Standard RNN updates are replaced by operations confined strictly to the Hyperboloid, scaling inputs by distance preserving transformations.

### 3.2 Liquid-MoE Synthesis
Instead of routing tokens through statically loaded expert parameters, the model dynamically synthesizes local, low-rank expert matrices on the fly using a conditioned hypernetwork.

### 3.3 Router Entropy Regularization
To prevent expert collapse, the system applies a continuous entropy loss penalty, forcing the synthesizer to distribute routing decisions uniformly.

### 3.4 Learnable Curvature
The layers initiate with near point-zero (Euclidean) curvature to stabilize gradient flow, while progressively "learning" higher curvature values at deeper layers to exponentially increase representational capacity.

## 4. Stability Engineering
Computing geometrically pure hyperbolic states initially induces massive floating-point errors (NaNs) due to the variance of exponential mapping. Hyper-SSM implements exact Riemannian structural bounds:

*   **Riemannian Clipping:** Constraining gradient norms aggressively without altering directional semantics.
*   **Stable `sinhc` Exp-Map:** Applying Taylor-series fallback around zero-points during projection to prevent division by zero.
*   **Spectral Normalization:** Capping explosive hypernetwork generated state variances.
*   **BF16 Precision Handling:** Relying completely on the expanded 8-bit exponent of `bfloat16` natively inside custom CUDA (`hyper_ssm_csrc`) threads to absorb hyperbolic variance.

## 5. Experiments
### Language Modeling
Initial runs over 40-Million parameter scales verified stable loss reduction and PyTorch gradient convergence without numerical collapse. 

### Multimodal Compatibility (Vision/Audio Transfer)
Continuous visual arrays (196 discrete patches) and Audio Mel-Spectrograms (80-bins) mapped natively to the Lorentzian topology directly without structural fault, establishing strong modality agnosticism.

### Embedded C Firmware Verification
To ascertain grammatical and structural sequence adoption beyond natural language distributions, a 40-Million parameter configuration was subjected to an extensive 3-epoch autoregressive continuous training run over an unadulterated 10.5M token corpus constructed directly from the FreeRTOS and ESP-IDF peripheral hardware abstraction layers. The system exhibited a seamless monotonic loss reduction trajectory ($10.99 \rightarrow 0.59$) without gradient explosion, effectively validating its capability to retain strict logical syntax inside hyperbolic state variables.

### Extreme Scaling on Constrained Hardware 
A primary claim of Hyper-SSM's utility resides in decoupling parameter volume from continuous context memory limits. By geometrically scaling the structural definitions to **~800 Million parameters** ($d=1152$, $L=36$), and leveraging dynamic state activation checkpointing, the system successfully executed forward and backward evaluation steps on an extreme consumer bottleneck constraint (6.00 GB NVIDIA RTX 4050), cresting precisely at $5.22$ GB peak allocation without throwing Out-of-Memory faults.

## 6. Limitations
While structurally sound, the architecture has explicit constraints preventing sweeping theoretical claims:
*   **Billion-Parameter Equivalence is Unproven:** Empirical scaling curves across experts ($E=4, E=16$) have not definitively proven that synthesized parameters mimic dense billion-parameter logic in practice.
*   **Precision Constraints:** Strict reliance on `float32` or `bfloat16`; native `float16` mixed-precision collapses immediately due to restricted exponent domains.
*   **Routing Overhead:** Generating matrices dynamically induces non-trivial CUDA computational latency over static memory lookups.

## 7. Conclusion
Hyper-SSM is a rigorously stabilized experimental geometric recurrent MoE architecture. By addressing the severe continuous numerical collapse associated with non-Euclidean state modifications, the model provides an O(1) state alternative to the conventional Transformer. 
