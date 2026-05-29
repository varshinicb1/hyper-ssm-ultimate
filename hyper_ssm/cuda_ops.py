"""
CUDA-accelerated Riemannian operations for Hyper-SSM.

This module provides a clean, centralized interface to the compiled
hyper_ssm_cuda extension (built from hyper_ssm/csrc/).

As of 2026 research (HiM, Nemotron hybrids, etc.), having a first-class
accelerated path for the geometric core is essential for competitiveness.

Usage:
    from hyper_ssm.cuda_ops import get_lorentz_product, get_project_to_tangent, is_cuda_available

    lorentz_fn = get_lorentz_product()
    proj_fn    = get_project_to_tangent()
"""

import torch
import warnings

_CUDA_EXTENSION = None
_CUDA_AVAILABLE = False

def _try_load_cuda_extension():
    global _CUDA_EXTENSION, _CUDA_AVAILABLE
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    try:
        import hyper_ssm_cuda
        _CUDA_EXTENSION = hyper_ssm_cuda
        _CUDA_AVAILABLE = True
        print("[hyper_ssm.cuda_ops] Successfully loaded compiled CUDA extension (hyper_ssm_cuda).")
    except ImportError as e:
        _CUDA_AVAILABLE = False
        warnings.warn(
            "[hyper_ssm.cuda_ops] CUDA extension 'hyper_ssm_cuda' not found. "
            "Falling back to pure PyTorch implementations. "
            "Build it with: python compile_cuda.py build_ext --inplace\n"
            f"Import error: {e}"
        )
    return _CUDA_EXTENSION


def is_cuda_available() -> bool:
    """Returns True if the compiled hyper_ssm_cuda extension is usable."""
    _try_load_cuda_extension()
    return _CUDA_AVAILABLE


def get_lorentz_product():
    """
    Returns a function that computes the Lorentz (Minkowski) inner product:
        <x, y>_L = -x0*y0 + sum(x[1:]*y[1:])
    Signature: (x: Tensor, y: Tensor) -> Tensor
    """
    ext = _try_load_cuda_extension()
    if _CUDA_AVAILABLE and hasattr(ext, "lorentz_product"):
        def _cuda_lorentz(x, y):
            # Extension expects contiguous tensors; make sure
            return ext.lorentz_product(x.contiguous(), y.contiguous())
        return _cuda_lorentz

    # High-quality pure PyTorch fallback (matches training scripts)
    def _py_lorentz(x, y):
        # x, y: [..., dim+1]
        m = x * y
        # -x0*y0 + sum_{i>=1}
        return torch.sum(m, dim=-1, keepdim=True) - 2 * m[..., 0:1]

    return _py_lorentz


def get_project_to_tangent():
    """
    Returns a function for Riemannian gradient projection onto the tangent space:
        out = g + <x, g>_L * x
    This is the correct way to project Euclidean gradients for Lorentz manifold optimization.
    """
    ext = _try_load_cuda_extension()
    if _CUDA_AVAILABLE and hasattr(ext, "project_to_tangent"):
        def _cuda_project(x, g):
            return ext.project_to_tangent(x.contiguous(), g.contiguous())
        return _cuda_project

    # Stable pure-PyTorch version used in the best training runs
    lorentz = get_lorentz_product()

    def _py_project(x, g):
        ip = lorentz(x, g)                    # [..., 1]
        ip = torch.clamp(ip, -10.0, 10.0)     # safety (matches train_c_code.py)
        result = g + ip * x
        # Kill NaNs aggressively (common in early hyperbolic training)
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        return result

    return _py_project


# Convenience: expose the raw module if someone needs lower-level access
def get_raw_cuda_module():
    _try_load_cuda_extension()
    return _CUDA_EXTENSION
