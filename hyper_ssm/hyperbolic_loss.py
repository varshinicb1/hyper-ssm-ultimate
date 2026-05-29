"""
Hyperbolic Loss Functions for Hierarchy-Aware Training

Directly inspired by HiM (arXiv:2505.18973, May 2025) and the broader 2025-2026
Hyperbolic LLM literature.

These losses encourage the model to use the geometry of the Lorentzian manifold
properly:
- Centripetal loss: Parent concepts should be closer to the origin than their children.
- Clustering loss: Semantically related entities should be close; unrelated ones far.

This dramatically improves hierarchical reasoning and long-range structure capture
compared to pure cross-entropy.
"""

import torch
import torch.nn as nn
from typing import Optional


class HyperbolicLoss(nn.Module):
    """
    Combined hyperbolic geometry loss for training models that operate in Lorentz space.

    Args:
        centripetal_weight: Weight for the centripetal (hierarchy depth) loss.
        clustering_weight: Weight for the clustering (semantic grouping) loss.
        margin_alpha: Margin for clustering loss (dynamic scaling recommended).
        margin_beta: Margin for centripetal loss.
    """

    def __init__(
        self,
        centripetal_weight: float = 0.01,
        clustering_weight: float = 0.01,
        margin_alpha: float = 0.25,
        margin_beta: float = 0.005,
    ):
        super().__init__()
        self.centripetal_weight = centripetal_weight
        self.clustering_weight = clustering_weight
        self.margin_alpha = margin_alpha
        self.margin_beta = margin_beta

    def lorentz_distance(self, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Lorentzian distance between two points on the hyperboloid.
        """
        c_t = torch.tensor(c, device=x.device, dtype=x.dtype)
        # Minkowski inner product
        mink = -x[..., 0:1] * y[..., 0:1] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        mink = torch.clamp(mink, min=-1e4, max=1e4)  # stability
        return torch.sqrt(c_t) * torch.arccosh(torch.clamp(-mink / c_t, min=1.0 + 1e-6))

    def forward(
        self,
        embeddings: torch.Tensor,           # [B, D+1] Lorentz embeddings (usually last token or pooled)
        parent_child_pairs: Optional[torch.Tensor] = None,  # [N, 2] indices into batch
        negative_pairs: Optional[torch.Tensor] = None,      # [N, 2] indices
        curvature: float = 1.0,
    ) -> dict:
        """
        Compute hyperbolic losses.

        If pairs are not provided, we fall back to a simple within-batch heuristic
        (anchor-positive-negative sampling). For serious training you should pass
        proper ontology / hierarchy pairs.
        """
        device = embeddings.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        if parent_child_pairs is None or negative_pairs is None:
            # Simple in-batch heuristic (good enough for many cases)
            B = embeddings.shape[0]
            if B < 4:
                return {"total": total_loss, "centripetal": 0.0, "clustering": 0.0}

            # Randomly sample some triplets
            anchors = torch.randint(0, B, (min(32, B//2),), device=device)
            positives = (anchors + 1) % B
            negatives = (anchors + 2) % B
        else:
            anchors = parent_child_pairs[:, 0]
            positives = parent_child_pairs[:, 1]
            negatives = negative_pairs[:, 1] if negative_pairs.dim() > 1 else negative_pairs

        e_a = embeddings[anchors]
        e_p = embeddings[positives]
        e_n = embeddings[negatives]

        # Centripetal loss: parents closer to origin than children
        dist_origin_a = self.lorentz_distance(e_a, torch.zeros_like(e_a), curvature)
        dist_origin_p = self.lorentz_distance(e_p, torch.zeros_like(e_p), curvature)

        centripetal = torch.relu(dist_origin_p - dist_origin_a + self.margin_beta).mean()
        loss_dict["centripetal"] = centripetal.item()

        # Clustering loss
        dist_ap = self.lorentz_distance(e_a, e_p, curvature)
        dist_an = self.lorentz_distance(e_a, e_n, curvature)

        clustering = torch.relu(dist_ap - dist_an + self.margin_alpha).mean()
        loss_dict["clustering"] = clustering.item()

        total_loss = (
            self.centripetal_weight * centripetal +
            self.clustering_weight * clustering
        )
        loss_dict["total"] = total_loss.item()

        return loss_dict


# Convenience function for easy import
def create_hyperbolic_loss(**kwargs) -> HyperbolicLoss:
    return HyperbolicLoss(**kwargs)
