"""Masked autoencoder architecture from Germain et al. (2015)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from made_reproduction.config import ActivationName


class MaskedLinear(nn.Linear):
    """Linear layer whose forbidden connections remain exactly zero."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    @torch.no_grad()
    def set_mask(self, mask: Tensor) -> None:
        if mask.shape != self.mask.shape:
            raise ValueError(
                f"expected mask shape {tuple(self.mask.shape)}, got {tuple(mask.shape)}"
            )
        self.mask.copy_(mask.to(device=self.mask.device, dtype=self.mask.dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Bernoulli MADE with reproducible order and connectivity masks.

    Input degrees are a permutation of ``1..D``. Hidden degrees lie between
    the smallest preceding degree and ``D-1``. Hidden connections use ``<=``;
    output and direct input-output connections use ``<``. Consequently output
    ``d`` can only depend on variables preceding it in the sampled ordering.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (500,),
        *,
        activation: ActivationName | str = ActivationName.TANH,
        direct_input_to_output: bool = True,
        hidden_to_output_skips: bool = False,
        zero_init_final_branch: bool = False,
        mask_seed: int = 1234,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("MADE requires at least two input dimensions")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation_name = ActivationName(activation)
        self.direct_input_to_output = direct_input_to_output
        self.hidden_to_output_skips = hidden_to_output_skips
        self.zero_init_final_branch = zero_init_final_branch
        self.mask_seed = mask_seed

        dimensions = (input_dim, *self.hidden_dims, input_dim)
        self.layers = nn.ModuleList(
            MaskedLinear(in_features, out_features)
            for in_features, out_features in zip(dimensions[:-1], dimensions[1:], strict=True)
        )
        self.hidden_skips = nn.ModuleList(
            MaskedLinear(width, input_dim, bias=False)
            for width in self.hidden_dims[:-1]
        ) if hidden_to_output_skips else nn.ModuleList()
        self.direct = (
            MaskedLinear(input_dim, input_dim, bias=False) if direct_input_to_output else None
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        for index, width in enumerate(self.hidden_dims):
            self.register_buffer(f"hidden_degrees_{index}", torch.zeros(width, dtype=torch.long))

        self.reset_parameters()
        self.set_masks(0)

    def reset_parameters(self) -> None:
        """Use the paper's orthogonal weight family and zero biases."""

        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.direct is not None:
            nn.init.orthogonal_(self.direct.weight)
        for layer in self.hidden_skips:
            nn.init.orthogonal_(layer.weight)
        if self.zero_init_final_branch and self.hidden_skips:
            nn.init.zeros_(self.layers[-1].weight)

    def _activation(self, values: Tensor) -> Tensor:
        if self.activation_name is ActivationName.TANH:
            return torch.tanh(values)
        return F.relu(values)

    @torch.no_grad()
    def set_masks(self, index: int) -> None:
        """Select a mask deterministically from ``mask_seed`` and ``index``."""

        if index < 0:
            raise ValueError("mask index must be non-negative")
        generator = torch.Generator(device="cpu").manual_seed(self.mask_seed + index)
        input_degrees = torch.randperm(self.input_dim, generator=generator) + 1
        degrees: list[Tensor] = [input_degrees]
        minimum_degree = 1
        for hidden_index, width in enumerate(self.hidden_dims):
            hidden_degrees = torch.randint(
                low=minimum_degree,
                high=self.input_dim,
                size=(width,),
                generator=generator,
            )
            degrees.append(hidden_degrees)
            minimum_degree = int(hidden_degrees.min())
            getattr(self, f"hidden_degrees_{hidden_index}").copy_(hidden_degrees)
        output_degrees = input_degrees

        for layer_index, layer in enumerate(self.layers):
            incoming = degrees[layer_index]
            outgoing = (
                output_degrees if layer_index == len(self.layers) - 1 else degrees[layer_index + 1]
            )
            if layer_index == len(self.layers) - 1:
                mask = outgoing[:, None] > incoming[None, :]
            else:
                mask = outgoing[:, None] >= incoming[None, :]
            layer.set_mask(mask)
        if self.direct is not None:
            self.direct.set_mask(output_degrees[:, None] > input_degrees[None, :])
        for hidden_index, layer in enumerate(self.hidden_skips):
            layer.set_mask(output_degrees[:, None] > degrees[hidden_index + 1][None, :])
        self.input_degrees.copy_(input_degrees)
        self.mask_index.fill_(index)

    @torch.no_grad()
    def resample_masks(self) -> None:
        self.set_masks(int(self.mask_index.item()) + 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        hidden_values = []
        for layer in self.layers[:-1]:
            hidden = self._activation(layer(hidden))
            hidden_values.append(hidden)
        logits = self.layers[-1](hidden)
        for layer, values in zip(self.hidden_skips, hidden_values, strict=False):
            logits = logits + layer(values)
        if self.direct is not None:
            logits = logits + self.direct(inputs)
        return logits

    def log_prob(self, inputs: Tensor) -> Tensor:
        """Return exact per-observation Bernoulli log likelihood in nats."""

        return -F.binary_cross_entropy_with_logits(
            self(inputs),
            inputs,
            reduction="none",
        ).sum(dim=-1)

    def ensemble_log_prob(self, inputs: Tensor, masks: int, *, start_index: int) -> Tensor:
        """Average model probabilities over deterministic MADE masks."""

        if masks < 1 or start_index < 0:
            raise ValueError("masks must be positive and start_index non-negative")
        original_index = int(self.mask_index.item())
        log_probabilities = []
        try:
            for index in range(start_index, start_index + masks):
                self.set_masks(index)
                log_probabilities.append(self.log_prob(inputs))
        finally:
            self.set_masks(original_index)
        stacked = torch.stack(log_probabilities, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(masks)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw ancestral binary samples using the current mask ordering."""

        if count < 1:
            raise ValueError("count must be positive")
        samples = torch.zeros(count, self.input_dim, device=self.input_degrees.device)
        for variable in torch.argsort(self.input_degrees):
            probabilities = torch.sigmoid(self(samples)[:, variable])
            samples[:, variable] = torch.bernoulli(probabilities, generator=generator)
        return samples


class ResidualMADE(nn.Module):
    """A degree-preserving residual MADE for the Part 2 architecture study.

    Every residual block maps hidden features of degree ``m`` to features of
    the same degree. Its masked branch only reads degrees at most ``m`` and
    its identity branch reads exactly degree ``m``. Thus neither branch can
    introduce a forbidden dependence into an output conditional.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_blocks: int,
        *,
        activation: ActivationName | str = ActivationName.RELU,
        direct_input_to_output: bool = True,
        mask_seed: int = 1234,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("MADE requires at least two input dimensions")
        if hidden_dim < 1 or residual_blocks < 1:
            raise ValueError("hidden_dim and residual_blocks must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual_blocks_count = residual_blocks
        self.activation_name = ActivationName(activation)
        self.direct_input_to_output = direct_input_to_output
        self.mask_seed = mask_seed

        self.input_layer = MaskedLinear(input_dim, hidden_dim)
        self.residual_layers = nn.ModuleList(
            MaskedLinear(hidden_dim, hidden_dim) for _ in range(residual_blocks)
        )
        self.output_layer = MaskedLinear(hidden_dim, input_dim)
        self.direct = (
            MaskedLinear(input_dim, input_dim, bias=False) if direct_input_to_output else None
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("hidden_degrees", torch.zeros(hidden_dim, dtype=torch.long))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))

        self.reset_parameters()
        self.set_masks(0)

    def reset_parameters(self) -> None:
        """Use the same orthogonal initialization family as paper MADE."""

        for layer in (self.input_layer, *self.residual_layers, self.output_layer):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.direct is not None:
            nn.init.orthogonal_(self.direct.weight)

    def _activation(self, values: Tensor) -> Tensor:
        if self.activation_name is ActivationName.TANH:
            return torch.tanh(values)
        return F.relu(values)

    @torch.no_grad()
    def set_masks(self, index: int) -> None:
        """Set deterministic degrees and masks without relaxing causality."""

        if index < 0:
            raise ValueError("mask index must be non-negative")
        generator = torch.Generator(device="cpu").manual_seed(self.mask_seed + index)
        input_degrees = torch.randperm(self.input_dim, generator=generator) + 1
        hidden_degrees = torch.randint(
            low=1,
            high=self.input_dim,
            size=(self.hidden_dim,),
            generator=generator,
        )
        self.input_layer.set_mask(hidden_degrees[:, None] >= input_degrees[None, :])
        # A residual layer preserves output degrees. Its transformed branch can
        # read only equal-or-earlier hidden degrees; the identity skip is exact.
        residual_mask = hidden_degrees[:, None] >= hidden_degrees[None, :]
        for layer in self.residual_layers:
            layer.set_mask(residual_mask)
        self.output_layer.set_mask(input_degrees[:, None] > hidden_degrees[None, :])
        if self.direct is not None:
            self.direct.set_mask(input_degrees[:, None] > input_degrees[None, :])
        self.input_degrees.copy_(input_degrees)
        self.hidden_degrees.copy_(hidden_degrees)
        self.mask_index.fill_(index)

    @torch.no_grad()
    def resample_masks(self) -> None:
        self.set_masks(int(self.mask_index.item()) + 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self._activation(self.input_layer(inputs))
        for layer in self.residual_layers:
            hidden = self._activation(hidden + layer(hidden))
        logits = self.output_layer(hidden)
        if self.direct is not None:
            logits = logits + self.direct(inputs)
        return logits

    def log_prob(self, inputs: Tensor) -> Tensor:
        return -F.binary_cross_entropy_with_logits(
            self(inputs), inputs, reduction="none"
        ).sum(dim=-1)

    def ensemble_log_prob(self, inputs: Tensor, masks: int, *, start_index: int) -> Tensor:
        if masks < 1 or start_index < 0:
            raise ValueError("masks must be positive and start_index non-negative")
        original_index = int(self.mask_index.item())
        log_probabilities = []
        try:
            for index in range(start_index, start_index + masks):
                self.set_masks(index)
                log_probabilities.append(self.log_prob(inputs))
        finally:
            self.set_masks(original_index)
        stacked = torch.stack(log_probabilities, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(masks)

    @torch.no_grad()
    def sample(
        self, count: int, *, generator: torch.Generator | None = None
    ) -> Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        samples = torch.zeros(count, self.input_dim, device=self.input_degrees.device)
        for variable in torch.argsort(self.input_degrees):
            probabilities = torch.sigmoid(self(samples)[:, variable])
            samples[:, variable] = torch.bernoulli(probabilities, generator=generator)
        return samples


class LocallyMaskedConv2d(nn.Module):
    """Shared convolution weights with a causal mask at every image location."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size * kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("mask", torch.empty(0))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def set_mask(self, degrees: Tensor, *, strict: bool) -> None:
        side = math.isqrt(degrees.numel())
        if side * side != degrees.numel():
            raise ValueError("locally masked convolution requires a square image")
        field = degrees.float().reshape(1, 1, side, side)
        padding = self.kernel_size // 2
        patches = F.unfold(field, self.kernel_size, padding=padding).squeeze(0)
        valid = F.unfold(
            torch.ones_like(field), self.kernel_size, padding=padding
        ).squeeze(0).bool()
        targets = degrees.reshape(1, -1)
        allowed = patches < targets if strict else patches <= targets
        self.mask = (allowed & valid).to(dtype=self.weight.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        patches = F.unfold(
            inputs, self.kernel_size, padding=self.kernel_size // 2
        ).reshape(
            inputs.shape[0], self.in_channels, self.kernel_size**2, -1
        )
        patches = patches * self.mask[None, None, :, :]
        outputs = torch.einsum("bckl,ock->bol", patches, self.weight)
        outputs = outputs + self.bias[None, :, None]
        return outputs.reshape(
            inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]
        )


class LocallyMaskedConvMADE(nn.Module):
    """Spatial MADE using arbitrary-order locally masked convolutions."""

    def __init__(
        self,
        input_dim: int,
        *,
        channels: int = 64,
        residual_blocks: int = 3,
        mask_seed: int = 1234,
        direct_input_to_output: bool = True,
    ):
        super().__init__()
        side = math.isqrt(input_dim)
        if side * side != input_dim:
            raise ValueError("LMConv MADE requires a square image")
        if channels < 1 or residual_blocks < 1:
            raise ValueError("channels and residual_blocks must be positive")
        self.input_dim = input_dim
        self.side = side
        self.mask_seed = mask_seed
        self.input_layer = LocallyMaskedConv2d(1, channels, 7)
        # Once the strict first layer has constructed a causal feature at each
        # pixel, 1x1 residual blocks can transform channels at that same degree
        # without opening any new spatial dependency. This is much cheaper than
        # repeating im2col-style locally masked convolutions on Apple MPS.
        self.residual_layers = nn.ModuleList(
            nn.Conv2d(channels, channels, 1) for _ in range(residual_blocks)
        )
        self.output_layer = nn.Conv2d(channels, 1, 1)
        self.direct = (
            MaskedLinear(input_dim, input_dim, bias=False)
            if direct_input_to_output
            else None
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        if self.direct is not None:
            nn.init.orthogonal_(self.direct.weight)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        self.set_masks(0)

    @torch.no_grad()
    def set_masks(self, index: int) -> None:
        if index < 0:
            raise ValueError("mask index must be non-negative")
        generator = torch.Generator(device="cpu").manual_seed(self.mask_seed + index)
        degrees = torch.randperm(self.input_dim, generator=generator) + 1
        self.input_layer.set_mask(degrees, strict=True)
        if self.direct is not None:
            self.direct.set_mask(degrees[:, None] > degrees[None, :])
        self.input_degrees.copy_(degrees)
        self.mask_index.fill_(index)

    @torch.no_grad()
    def resample_masks(self) -> None:
        self.set_masks(int(self.mask_index.item()) + 1)

    def forward(self, inputs: Tensor) -> Tensor:
        image = inputs.reshape(-1, 1, self.side, self.side)
        hidden = F.relu(self.input_layer(image))
        for layer in self.residual_layers:
            hidden = F.relu(hidden + layer(hidden))
        logits = self.output_layer(hidden).flatten(1)
        if self.direct is not None:
            logits = logits + self.direct(inputs)
        return logits

    def log_prob(self, inputs: Tensor) -> Tensor:
        return -F.binary_cross_entropy_with_logits(
            self(inputs), inputs, reduction="none"
        ).sum(dim=-1)

    def ensemble_log_prob(self, inputs: Tensor, masks: int, *, start_index: int) -> Tensor:
        if masks != 1:
            raise ValueError("LMConv candidate currently uses one fixed ordering")
        del start_index
        return self.log_prob(inputs)

    @torch.no_grad()
    def sample(
        self, count: int, *, generator: torch.Generator | None = None
    ) -> Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        samples = torch.zeros(count, self.input_dim, device=self.input_degrees.device)
        for variable in torch.argsort(self.input_degrees):
            probabilities = torch.sigmoid(self(samples)[:, variable])
            samples[:, variable] = torch.bernoulli(probabilities, generator=generator)
        return samples


class RasterMaskedConv2d(nn.Conv2d):
    """PixelCNN convolution with a raster-order type-A or type-B mask."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        include_center: bool,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        mask = torch.ones_like(self.weight)
        center = kernel_size // 2
        mask[:, :, center + 1 :, :] = 0
        mask[:, :, center, center + int(include_center) :] = 0
        self.register_buffer("causal_mask", mask)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.weight * self.causal_mask,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class PixelCNNMADE(nn.Module):
    """Compact residual PixelCNN for binary MNIST likelihoods."""

    def __init__(
        self,
        input_dim: int,
        *,
        channels: int = 32,
        residual_blocks: int = 4,
        direct_input_to_output: bool = True,
    ):
        super().__init__()
        side = math.isqrt(input_dim)
        if side * side != input_dim:
            raise ValueError("PixelCNN MADE requires a square image")
        self.input_dim = input_dim
        self.side = side
        self.input_layer = RasterMaskedConv2d(
            1, channels, 7, include_center=False
        )
        self.residual_layers = nn.ModuleList(
            RasterMaskedConv2d(
                channels, channels, 3, include_center=True
            )
            for _ in range(residual_blocks)
        )
        self.output_layer = nn.Conv2d(channels, 1, 1)
        self.direct = (
            MaskedLinear(input_dim, input_dim, bias=False)
            if direct_input_to_output
            else None
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        if self.direct is not None:
            self.direct.set_mask(
                self.input_degrees[:, None] > self.input_degrees[None, :]
            )
            nn.init.orthogonal_(self.direct.weight)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def set_masks(self, index: int) -> None:
        if index != 0:
            raise ValueError("PixelCNN uses one fixed raster ordering")

    def resample_masks(self) -> None:
        raise ValueError("PixelCNN uses one fixed raster ordering")

    def forward(self, inputs: Tensor) -> Tensor:
        image = inputs.reshape(-1, 1, self.side, self.side)
        hidden = F.relu(self.input_layer(image))
        for layer in self.residual_layers:
            hidden = F.relu(hidden + layer(hidden))
        logits = self.output_layer(hidden).flatten(1)
        if self.direct is not None:
            logits = logits + self.direct(inputs)
        return logits

    def log_prob(self, inputs: Tensor) -> Tensor:
        return -F.binary_cross_entropy_with_logits(
            self(inputs), inputs, reduction="none"
        ).sum(dim=-1)

    def ensemble_log_prob(self, inputs: Tensor, masks: int, *, start_index: int) -> Tensor:
        if masks != 1:
            raise ValueError("PixelCNN uses one fixed raster ordering")
        del start_index
        return self.log_prob(inputs)

    @torch.no_grad()
    def sample(
        self, count: int, *, generator: torch.Generator | None = None
    ) -> Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        samples = torch.zeros(count, self.input_dim, device=self.input_degrees.device)
        for variable in range(self.input_dim):
            probabilities = torch.sigmoid(self(samples)[:, variable])
            samples[:, variable] = torch.bernoulli(probabilities, generator=generator)
        return samples
