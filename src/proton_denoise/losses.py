from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsWeightedMSELoss(nn.Module):
    """DeepMC-style exponentially weighted MSE using target and prediction."""

    def __init__(self, alpha: float = 3.0, min_weight: float | None = None, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.min_weight = min_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Eq-style weighting: exp(-alpha * (1 - 0.5*(Y + Yhat)/max(Y))).
        y_max = torch.amax(target, dim=(-3, -2, -1), keepdim=True)
        y_max = torch.clamp(y_max, min=self.eps)
        y_avg_norm = 0.5 * (target + pred) / y_max
        y_avg_norm = torch.clamp(y_avg_norm, min=0.0, max=1.0)

        weights = torch.exp(-self.alpha * (1.0 - y_avg_norm))
        if self.min_weight is not None:
            weights = torch.clamp(weights, min=self.min_weight, max=1.0)
        return torch.mean(weights * (pred - target) ** 2)
