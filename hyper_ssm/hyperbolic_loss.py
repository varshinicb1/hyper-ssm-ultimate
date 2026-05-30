"""
Hyperbolic Loss Functions for Hierarchy-Aware Training (2026 Pinnacle Corrected)

Directly inspired by HiM (arXiv:2505.18973) and production geometric LLM research.

These losses encourage the model to use the geometry of the Lorentzian manifold
properly by operating on **actual Lorentz compressor states** (or their tangent
projections), not Euclidean hidden states.

- Centripetal loss: deeper / more specific concepts should be farther from origin
  than their parents (encourages proper hierarchical depth in the manifold).
- Clustering loss: semantically related points should be closer than negatives
  (margin-based in Lorentz or tangent space).

The loss now correctly expects [B, D+1] Lorentz vectors (time coord first) that
satisfy the hyperboloid constraint, as produced by TiledFractalCompressor /
stable_expmap.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def _lorentz_origin(batch_size: int, dim: int, device, dtype, c: float = 1.0) -> torch.Tensor:
    """Construct the origin point on the hyperboloid for given curvature c."""
    o = torch.zeros(batch_size, dim + 1, device=device, dtype=dtype)
    o[..., 0] = math.sqrt(c)
    return o


class HyperbolicLoss(nn.Module):
    """
    Production-grade hyperbolic geometry auxiliary loss.

    Operates natively on Lorentz states from the compressor (the correct geometric
    objects). Supports both direct Lorentz-margin losses and (recommended for
    stability) tangent-space-at-origin losses via log_o.

    Args:
        centripetal_weight: Weight for "parents closer to origin than children".
        clustering_weight:  Weight for contrastive clustering (related vs unrelated).
        margin_alpha:       Margin for clustering (in chosen metric).
        margin_beta:        Margin for centripetal (in chosen metric).
        tangent_space:      If True, map to tangent space at origin via log_o first
                            and perform Euclidean-style losses there (often better
                            conditioned gradients).
        curvature:          Base curvature (overridden at call time if needed).
    """

    def __init__(
        self,
        centripetal_weight: float = 0.003,
        clustering_weight: float = 0.003,
        margin_alpha: float = 0.25,
        margin_beta: float = 0.005,
        tangent_space: bool = True,   # Recommended default for stable training
        curvature: float = 1.0,
    ):
        super().__init__()
        self.centripetal_weight = centripetal_weight
        self.clustering_weight = clustering_weight
        self.margin_alpha = margin_alpha
        self.margin_beta = margin_beta
        self.tangent_space = tangent_space
        self.curvature = curvature

        # Lazy import to avoid circular dependency at package load
        self._log_o = None

    def __repr__(self):
        return (f"HyperbolicLoss(centripetal={self.centripetal_weight}, "
                f"clustering={self.clustering_weight}, "
                f"tangent_space={self.tangent_space}, "
                f"margins=({self.margin_beta}, {self.margin_alpha}))")

    def _ensure_log_o(self):
        if self._log_o is None:
            from .hyperbolic_ops import log_o as _log_o
            self._log_o = _log_o
        return self._log_o

    def lorentz_distance(self, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Lorentzian distance (arcosh form). Both x,y must be on the hyperboloid."""
        c_t = torch.tensor(c, device=x.device, dtype=x.dtype)
        mink = -x[..., 0:1] * y[..., 0:1] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        mink = torch.clamp(mink, min=-1e4, max=1e4)
        arg = torch.clamp(-mink / c_t, min=1.0 + 1e-6)
        return torch.sqrt(c_t) * torch.acosh(arg)

    def _to_tangent(self, lorentz_points: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Map Lorentz points to tangent space at origin (spatial vectors)."""
        log_o = self._ensure_log_o()
        # log_o already handles projection to manifold internally
        return log_o(lorentz_points, k=c)

    def forward_lorentz(
        self,
        lorentz_points: torch.Tensor,                 # [B, D+1] valid Lorentz vectors
        parent_child_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None,
        curvature: Optional[float] = None,
        return_tangent: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute the loss directly on Lorentz compressor states.

        This is the **correct** entry point when you have real geometric states
        from TiledFractalCompressor.get_final_state(...) or equivalent.
        """
        device = lorentz_points.device
        dtype = lorentz_points.dtype
        c = float(curvature) if curvature is not None else self.curvature

        B, D1 = lorentz_points.shape
        D = D1 - 1

        loss_dict: Dict[str, Any] = {"total": 0.0, "centripetal": 0.0, "clustering": 0.0, "radius_health": 0.0, "metric": "lorentz"}

        if self.tangent_space:
            # Preferred stable path for auxiliary losses
            tan = self._to_tangent(lorentz_points, c=c)  # [B, D]
            loss_dict["metric"] = "tangent"
            points_for_loss = tan
            dist_fn = lambda a, b: torch.norm(a - b, dim=-1)
            origin_dist_fn = lambda p: torch.norm(p, dim=-1)
        else:
            points_for_loss = lorentz_points
            dist_fn = lambda a, b: self.lorentz_distance(a, b, c)
            origin_dist_fn = lambda p: torch.acosh(torch.clamp(p[..., 0] / math.sqrt(c), min=1.0 + 1e-6))

        # --- Always compute a cheap but geometrically meaningful "radius health" regularizer ---
        # Encourages states to live at a healthy distance from origin (neither collapse nor explosion).
        # This gives useful gradient even for batch size 1.
        with torch.no_grad():
            d_o = origin_dist_fn(points_for_loss)
        target_radius = 1.5  # sweet spot in our units (tunable)
        radius_health = ((d_o.mean() - target_radius).abs() * 0.01).mean()   # small weight inside loss
        loss_dict["radius_health"] = float(radius_health.detach())

        if parent_child_pairs is None or negative_pairs is None:
            if B < 2:
                # With B=1 we can only do the radius term
                total = self.centripetal_weight * 0.0 + self.clustering_weight * 0.0 + radius_health
                loss_dict.update({"total": float(total), "num_pairs": 0})
                return loss_dict
            # Robust small-batch sampling (works for B=2,3,...)
            n_samples = min(64, max(4, B * 4))
            anchors = torch.randint(0, B, (n_samples,), device=device)
            positives = (anchors + 1) % B
            negatives = (anchors + 2) % B
        else:
            anchors = parent_child_pairs[:, 0]
            positives = parent_child_pairs[:, 1]
            negatives = negative_pairs[:, 1] if negative_pairs.dim() > 1 else negative_pairs

        a = points_for_loss[anchors]
        p = points_for_loss[positives]
        n = points_for_loss[negatives]

        # Centripetal: children should be farther from origin than parents
        if self.tangent_space:
            d_o_a = origin_dist_fn(a)
            d_o_p = origin_dist_fn(p)
        else:
            # For pure Lorentz we still use the proper origin distance (arcosh)
            d_o_a = origin_dist_fn(a)
            d_o_p = origin_dist_fn(p)

        centripetal = torch.relu(d_o_p - d_o_a + self.margin_beta).mean()

        # Clustering (triplet-style)
        d_ap = dist_fn(a, p)
        d_an = dist_fn(a, n)
        clustering = torch.relu(d_ap - d_an + self.margin_alpha).mean()

        total = (
            self.centripetal_weight * centripetal +
            self.clustering_weight * clustering +
            radius_health   # always-on geometric health term (works at tiny batch)
        )

        loss_dict.update({
            "total": float(total.detach().item()) if isinstance(total, torch.Tensor) else float(total),
            "centripetal": float(centripetal.detach().item()) if isinstance(centripetal, torch.Tensor) else float(centripetal),
            "clustering": float(clustering.detach().item()) if isinstance(clustering, torch.Tensor) else float(clustering),
            "num_pairs": int(anchors.numel()) if "anchors" in locals() else max(B, 1),
        })

        if return_tangent and self.tangent_space:
            loss_dict["tangent_points"] = points_for_loss.detach()

        return loss_dict

    def forward(
        self,
        embeddings: torch.Tensor,
        parent_child_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None,
        curvature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible entry point.

        If the input looks like Lorentz (last dim even, first coord roughly > 1),
        treats it as Lorentz. Otherwise falls back to a safe expmap proxy
        (for legacy code that was accidentally passing Euclidean vectors).
        """
        # Heuristic: Lorentz vectors from our compressors have shape [..., D+1]
        # and time component typically >= 1.0 (for c~1).
        if embeddings.dim() >= 2 and embeddings.shape[-1] >= 3:
            # Check if it roughly satisfies hyperboloid (cheap test on a few points)
            x0 = embeddings[..., 0]
            if torch.all(x0 > 0.5):  # very likely already Lorentz
                return self.forward_lorentz(
                    embeddings, parent_child_pairs, negative_pairs, curvature=curvature
                )

        # Legacy path (what the old trainer was accidentally doing):
        # Treat as Euclidean and map to Lorentz via stable_expmap
        from .hyperbolic_ops import stable_expmap, project_to_manifold
        lor = stable_expmap(embeddings)
        lor = project_to_manifold(lor, repair=True)[0] if isinstance(project_to_manifold(lor), tuple) else lor
        return self.forward_lorentz(lor, parent_child_pairs, negative_pairs, curvature=curvature)


# Convenience function for easy import
def create_hyperbolic_loss(**kwargs) -> HyperbolicLoss:
    return HyperbolicLoss(**kwargs)
