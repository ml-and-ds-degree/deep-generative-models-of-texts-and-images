"""Course-aligned sample metrics for binary MADE experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def unbiased_rbf_mmd(
    generated: Tensor,
    reference: Tensor,
    *,
    bandwidth_squared: float,
) -> Tensor:
    """Return unbiased squared MMD with a fixed Gaussian RBF kernel.

    The diagonal terms are excluded from the within-set averages, avoiding the
    finite-sample bias of the usual V-statistic.  The bandwidth is an explicit
    protocol choice rather than a value re-estimated for each model.
    """

    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("MMD inputs must have shape [examples, features]")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("MMD inputs must have the same feature dimension")
    if generated.shape[0] < 2 or reference.shape[0] < 2:
        raise ValueError("unbiased MMD requires at least two examples per set")
    if bandwidth_squared <= 0:
        raise ValueError("bandwidth_squared must be positive")

    generated = generated.float()
    reference = reference.float()
    scale = -0.5 / bandwidth_squared
    k_xx = torch.exp(scale * torch.cdist(generated, generated).square())
    k_yy = torch.exp(scale * torch.cdist(reference, reference).square())
    k_xy = torch.exp(scale * torch.cdist(generated, reference).square())
    count_x = generated.shape[0]
    count_y = reference.shape[0]
    within_x = (k_xx.sum() - k_xx.diagonal().sum()) / (count_x * (count_x - 1))
    within_y = (k_yy.sum() - k_yy.diagonal().sum()) / (count_y * (count_y - 1))
    return within_x + within_y - 2 * k_xy.mean()
