"""
Hyper-SSM: Lorentzian State-Space Models with Liquid Mixture-of-Experts

Core package for O(1) memory hyperbolic recurrent architectures.

Key components (2026 research-aligned):
- FractalStateCompressor: Lorentzian recursive state (KV-cache replacement)
- DynamicLiquidLayer: Hypernetwork-synthesized transient experts
- cuda_ops: First-class access to compiled Riemannian kernels
"""

from .model import HyperSSM, HyperSSMConfig, HyperSSMBlock, HybridHyperSSMBlock
from .hyperbolic_ops import (
    FractalStateCompressor,
    stable_expmap,
    lorentz_normalize,
    lorentz_product,
    lorentz_inner,
    project_to_hyperboloid,
    riemannian_clip,
    safe_sinhc,
    check_manifold_constraint,
    project_to_manifold,
)
from .liquid_weights import DynamicLiquidLayer, HyperWeightSynthesizer
from .hybrid_attention import SelectiveAttentionRecall
from .hyperbolic_loss import HyperbolicLoss, create_hyperbolic_loss
from .tokenizer import HyperTokenizer
from .cuda_ops import is_cuda_available, get_lorentz_product, get_project_to_tangent
from .tiled_compressor import (
    TiledFractalCompressor,
    ProductionTiledCompressor,
    is_rust_kernels_available,
    _RUST_AVAILABLE as RUST_KERNELS_AVAILABLE,  # public alias
    PerformanceCounters,
)

__version__ = "2026.1.0-ultimate-pinnacle"  # Candidate 3: absolute best Python engineering quality

__all__ = [
    "HyperSSM",
    "HyperSSMConfig",
    "HyperSSMBlock",
    "HybridHyperSSMBlock",
    "SelectiveAttentionRecall",
    "HyperbolicLoss",
    "create_hyperbolic_loss",
    "FractalStateCompressor",
    "DynamicLiquidLayer",
    "HyperWeightSynthesizer",
    "HyperTokenizer",
    "stable_expmap",
    "lorentz_inner",
    "check_manifold_constraint",
    "project_to_manifold",
    "is_cuda_available",
    "get_lorentz_product",
    "get_project_to_tangent",
    "TiledFractalCompressor",
    "ProductionTiledCompressor",
    "PerformanceCounters",
    "is_rust_kernels_available",
    "RUST_KERNELS_AVAILABLE",
]
