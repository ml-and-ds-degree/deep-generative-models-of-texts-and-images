"""Paper-specific metrics not already provided by TorchMetrics."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class ActiveUnits(Metric):
    """Count dimensions with Var_x(E_q[z|x]) > 0.01, as in the IWAE paper.

    This is a descriptive representation diagnostic, not evidence that a
    coordinate is disentangled. The selected K=50 paper result is 25 of 50
    active units; it is a reference value rather than a pass/fail threshold.
    """

    full_state_update = False

    def __init__(self, latent_dim: int, threshold: float = 1e-2):
        super().__init__()
        self.threshold = threshold
        self.add_state(
            "mean_sum",
            default=torch.zeros(latent_dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "mean_square_sum",
            default=torch.zeros(latent_dim),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros((), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, posterior_mean: Tensor) -> None:
        self.mean_sum += posterior_mean.sum(0)
        self.mean_square_sum += posterior_mean.square().sum(0)
        self.count += posterior_mean.shape[0]

    def compute(self) -> Tensor:
        count = self.count.clamp_min(1)
        variance = self.mean_square_sum / count - (self.mean_sum / count).square()
        return (variance > self.threshold).sum().to(torch.float32)
