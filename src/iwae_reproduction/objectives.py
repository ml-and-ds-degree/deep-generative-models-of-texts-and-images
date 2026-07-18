"""IWAE bounds and the doubly-reparameterized encoder surrogate.

PyTorch does not provide either research objective. They remain explicit so
their gradients can be checked against Burda et al. and Tucker et al. DReG
changes the encoder gradient estimator only; the generative objective and
decoder gradient remain the original IWAE ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from iwae_reproduction.networks import normal_log_prob


@dataclass(frozen=True, slots=True)
class ImportanceTerms:
    """Quantities shared by the IWAE and DReG objectives."""

    log_weights: Tensor
    pathwise_log_weights: Tensor


def importance_terms(
    x: Tensor,
    z: Tensor,
    mean: Tensor,
    log_std: Tensor,
    decoder_logits: Tensor,
) -> ImportanceTerms:
    """Calculate log p(x,z)-log q(z|x), including a DReG pathwise variant."""
    batch, particles, pixels = decoder_logits.shape
    targets = x[:, None, :].expand(batch, particles, pixels)
    log_px_given_z = -F.binary_cross_entropy_with_logits(
        decoder_logits, targets, reduction="none"
    ).sum(-1)
    zeros = torch.zeros((), device=z.device, dtype=z.dtype)
    log_pz = normal_log_prob(z, zeros, zeros)
    log_qz = normal_log_prob(z, mean[:, None, :], log_std[:, None, :])
    # DReG needs this separate path: stop the explicit score-function path
    # through q while retaining the reparameterized dz/dphi path.
    log_qz_pathwise = normal_log_prob(z, mean.detach()[:, None, :], log_std.detach()[:, None, :])
    return ImportanceTerms(
        log_weights=log_px_given_z + log_pz - log_qz,
        pathwise_log_weights=log_px_given_z + log_pz - log_qz_pathwise,
    )


def log_mean_weight(log_weights: Tensor) -> Tensor:
    """Per-example Monte Carlo estimate of log p(x)."""
    return torch.logsumexp(log_weights, dim=1) - math.log(log_weights.shape[1])


def iwae_loss(log_weights: Tensor) -> Tensor:
    """Negative importance-weighted evidence lower bound."""
    return -log_mean_weight(log_weights).mean()


def dreg_encoder_loss(log_weights: Tensor, pathwise_log_weights: Tensor) -> Tensor:
    """DReG surrogate whose gradient is the IWAE encoder gradient estimator."""
    normalized_weights = torch.softmax(log_weights, dim=1).detach()
    return -(normalized_weights.square() * pathwise_log_weights).sum(dim=1).mean()
