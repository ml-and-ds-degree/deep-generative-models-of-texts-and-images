"""Neural networks used in the one-stochastic-layer IWAE."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _initialize_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class GaussianEncoder(nn.Module):
    """784-200-200 encoder with a diagonal 50-dimensional Gaussian output."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 200, latent_dim: int = 50):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _initialize_linear(module)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(x)
        return self.mean(hidden), self.log_std(hidden)


class BernoulliDecoder(nn.Module):
    """50-200-200 decoder returning Bernoulli logits over 784 pixels."""

    def __init__(self, latent_dim: int = 50, hidden_dim: int = 200, output_dim: int = 784):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _initialize_linear(module)

    @property
    def output_layer(self) -> nn.Linear:
        return self.network[-1]  # type: ignore[return-value]

    @torch.no_grad()
    def initialize_output_bias(self, pixel_mean: Tensor) -> None:
        """Initialize p(x|z)'s bias to the empirical pixel log odds."""
        probabilities = pixel_mean.flatten().clamp(1e-3, 1 - 1e-3)
        self.output_layer.bias.copy_(torch.logit(probabilities))

    def forward(self, z: Tensor) -> Tensor:
        return self.network(z)


def sample_diagonal_gaussian(
    mean: Tensor,
    log_std: Tensor,
    particles: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw reparameterized samples with shape ``[batch, particles, latent]``."""
    epsilon = torch.randn(
        mean.shape[0],
        particles,
        mean.shape[-1],
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return mean[:, None, :] + epsilon * log_std.exp()[:, None, :]


def normal_log_prob(value: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Sum diagonal-Normal log probabilities over the final dimension."""
    standardized = (value - mean) * torch.exp(-log_std)
    return (-0.5 * standardized.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
