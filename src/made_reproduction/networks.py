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


class AttentionMADE(nn.Module):
    """MADE with residual LayerNorm Transformer blocks on the paper ordering.

    Inputs are permuted into the sampled MADE degree order, right-shifted so
    position ``t`` never sees ``x_t``, then encoded with ``TransformerEncoder``
    (pre-norm residual attention and MLP, dropout, final LayerNorm). Logits are
    scattered back to the original pixel layout. The causal mask is PyTorch's
    square subsequent mask; the only MADE-specific tensor is the degree
    permutation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_blocks: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask_seed: int = 1234,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError("MADE requires at least two input dimensions")
        if hidden_dim < 1 or residual_blocks < 1:
            raise ValueError("hidden_dim and residual_blocks must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual_blocks_count = residual_blocks
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.mask_seed = mask_seed

        self.value_proj = nn.Linear(1, hidden_dim)
        self.position_embed = nn.Embedding(input_dim, hidden_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=residual_blocks,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("generation_order", torch.arange(input_dim))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(input_dim),
        )
        nn.init.normal_(self.start_token, mean=0.0, std=0.02)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.set_masks(0)

    @torch.no_grad()
    def set_masks(self, index: int) -> None:
        """Sample the MADE input permutation; Transformer positions stay 0..D-1."""

        if index < 0:
            raise ValueError("mask index must be non-negative")
        generator = torch.Generator(device="cpu").manual_seed(self.mask_seed + index)
        input_degrees = torch.randperm(self.input_dim, generator=generator) + 1
        self.input_degrees.copy_(input_degrees)
        self.generation_order.copy_(torch.argsort(input_degrees))
        self.mask_index.fill_(index)

    @torch.no_grad()
    def resample_masks(self) -> None:
        self.set_masks(int(self.mask_index.item()) + 1)

    def _ordered_tokens(self, inputs: Tensor) -> Tensor:
        ordered = inputs[:, self.generation_order]
        previous = self.value_proj(ordered[:, :-1].unsqueeze(-1))
        start = self.start_token.expand(inputs.shape[0], 1, -1)
        positions = torch.arange(self.input_dim, device=inputs.device)
        return torch.cat([start, previous], dim=1) + self.position_embed(positions)

    def forward(self, inputs: Tensor) -> Tensor:
        tokens = self._ordered_tokens(inputs)
        hidden = self.encoder(tokens, mask=self.causal_mask)
        ordered_logits = self.output_head(hidden).squeeze(-1)
        logits = inputs.new_empty(inputs.shape)
        logits[:, self.generation_order] = ordered_logits
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
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        chunk_size: int = 16,
    ) -> Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        chunks = []
        for start in range(0, count, chunk_size):
            width = min(chunk_size, count - start)
            samples = torch.zeros(
                width, self.input_dim, device=self.input_degrees.device
            )
            for variable in self.generation_order:
                probabilities = torch.sigmoid(self(samples)[:, variable])
                samples[:, variable] = torch.bernoulli(
                    probabilities, generator=generator
                )
            chunks.append(samples)
        return torch.cat(chunks, dim=0)


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
        # Follow the weight's device: order-agnostic training re-sets masks
        # after the module has already been moved to MPS or CUDA.
        self.mask = (allowed & valid).to(
            dtype=self.weight.dtype, device=self.weight.device
        )

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


def s_curve_order_degrees(side: int, index: int) -> Tensor:
    """Return degrees ``1..side**2`` (row-major flattened) for one S-curve order.

    The base order is the boustrophedon scan (left-to-right on even rows,
    right-to-left on odd rows). ``index % 8`` selects one of the eight
    dihedral transforms of that curve, giving eight distinct generation
    orders that all traverse the image along locally contiguous paths.
    """

    positions = torch.arange(side * side).reshape(side, side)
    positions[1::2] = positions[1::2].flip(-1)
    if index % 8 & 1:
        positions = positions.flip(0)
    if index % 8 & 2:
        positions = positions.flip(1)
    if index % 8 & 4:
        positions = positions.transpose(0, 1)
    return (positions + 1).reshape(-1)


def _gated_activation(values: Tensor) -> Tensor:
    """The Gated PixelCNN unit: split channels, then tanh times sigmoid."""

    content, gate = values.chunk(2, dim=1)
    return torch.tanh(content) * torch.sigmoid(gate)


class VerticalStackConv2d(nn.Module):
    """Convolution whose output at row r sees input rows strictly above r."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        half = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (half + 1, kernel_size))
        # (left, right, top, bottom): padding the top by half+1 rows and
        # cropping the bottom shifts the window fully above the output row.
        self.padding = (half, half, half + 1, 0)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv(F.pad(inputs, self.padding))
        return outputs[:, :, : inputs.shape[2], :]


class HorizontalStackConv2d(nn.Module):
    """1-by-k convolution over pixels left of (type A) or up to (type B) each column."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        include_center: bool,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        width = kernel_size // 2 + 1
        self.conv = nn.Conv2d(in_channels, out_channels, (1, width))
        self.padding = (width - int(include_center), 0, 0, 0)
        self.include_center = include_center

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv(F.pad(inputs, self.padding))
        return outputs[:, :, :, : inputs.shape[3]]


class GatedPixelCNNLayer(nn.Module):
    """One vertical-plus-horizontal gated layer from van den Oord et al. (2016)."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int,
        *,
        include_center: bool,
        residual: bool,
    ):
        super().__init__()
        self.vertical = VerticalStackConv2d(in_channels, 2 * channels, kernel_size)
        self.vertical_to_horizontal = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.horizontal = HorizontalStackConv2d(
            in_channels, 2 * channels, kernel_size, include_center=include_center
        )
        self.horizontal_residual = nn.Conv2d(channels, channels, 1) if residual else None

    def forward(self, vertical: Tensor, horizontal: Tensor) -> tuple[Tensor, Tensor]:
        vertical_pre = self.vertical(vertical)
        horizontal_pre = self.horizontal(horizontal) + self.vertical_to_horizontal(
            vertical_pre
        )
        horizontal_out = _gated_activation(horizontal_pre)
        if self.horizontal_residual is not None:
            horizontal_out = horizontal + self.horizontal_residual(horizontal_out)
        return _gated_activation(vertical_pre), horizontal_out


class GatedPixelCNN(nn.Module):
    """Gated PixelCNN without the masked-convolution blind spot.

    The vertical stack conditions on all rows above the current pixel and the
    horizontal stack on earlier pixels of the current row, so unlike stacked
    single-mask convolutions the receptive field grows without a blind spot.
    Gated tanh-sigmoid units replace ReLU, and the horizontal stack carries
    residual connections. Raster order, exact Bernoulli likelihood.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        channels: int = 96,
        residual_blocks: int = 10,
        input_kernel_size: int = 7,
        kernel_size: int = 5,
    ):
        super().__init__()
        side = math.isqrt(input_dim)
        if side * side != input_dim:
            raise ValueError("Gated PixelCNN requires a square image")
        if channels < 1 or residual_blocks < 1:
            raise ValueError("channels and residual_blocks must be positive")
        self.input_dim = input_dim
        self.side = side
        self.input_layer = GatedPixelCNNLayer(
            1, channels, input_kernel_size, include_center=False, residual=False
        )
        self.layers = nn.ModuleList(
            GatedPixelCNNLayer(
                channels, channels, kernel_size, include_center=True, residual=True
            )
            for _ in range(residual_blocks)
        )
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def set_masks(self, index: int) -> None:
        if index != 0:
            raise ValueError("Gated PixelCNN uses one fixed raster ordering")

    def resample_masks(self) -> None:
        raise ValueError("Gated PixelCNN uses one fixed raster ordering")

    def forward(self, inputs: Tensor) -> Tensor:
        image = inputs.reshape(-1, 1, self.side, self.side)
        vertical, horizontal = self.input_layer(image, image)
        for layer in self.layers:
            vertical, horizontal = layer(vertical, horizontal)
        return self.output_head(horizontal).flatten(1)

    def log_prob(self, inputs: Tensor) -> Tensor:
        return -F.binary_cross_entropy_with_logits(
            self(inputs), inputs, reduction="none"
        ).sum(dim=-1)

    def ensemble_log_prob(self, inputs: Tensor, masks: int, *, start_index: int) -> Tensor:
        if masks != 1:
            raise ValueError("Gated PixelCNN uses one fixed raster ordering")
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


class GatedLocallyMaskedBlock(nn.Module):
    """Residual gated block built on a per-location masked 3-by-3 convolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = LocallyMaskedConv2d(channels, 2 * channels, kernel_size)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.projection(_gated_activation(self.conv(hidden)))


class OrderEnsembleLMConvMADE(nn.Module):
    """Order-agnostic locally masked convolutional MADE (Jain et al., 2020).

    MADE's two ideas are kept: weight masking gives exact one-pass
    autoregressive likelihoods, and averaging over orderings improves the
    density estimate. The masks are realized spatially with per-location
    masked convolutions, so unlike a shared raster weight mask there is no
    blind spot, and unlike dense MADE the parameters respect image geometry.
    ``set_masks(index)`` selects one of eight dihedral S-curve orders;
    training cycles them per batch and evaluation averages all eight.
    """

    ORDER_COUNT = 8

    def __init__(
        self,
        input_dim: int,
        *,
        channels: int = 64,
        residual_blocks: int = 8,
        input_kernel_size: int = 7,
        kernel_size: int = 3,
        mask_seed: int = 1234,
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
        self.input_layer = LocallyMaskedConv2d(1, 2 * channels, input_kernel_size)
        self.blocks = nn.ModuleList(
            GatedLocallyMaskedBlock(channels, kernel_size)
            for _ in range(residual_blocks)
        )
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.register_buffer("input_degrees", torch.arange(1, input_dim + 1))
        self.register_buffer("mask_index", torch.tensor(0, dtype=torch.long))
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        self.set_masks(0)

    @torch.no_grad()
    def set_masks(self, index: int) -> None:
        if index < 0:
            raise ValueError("mask index must be non-negative")
        degrees = s_curve_order_degrees(self.side, index)
        # The input layer is strict (excludes the current pixel); hidden
        # features at a location then depend only on strictly earlier pixels,
        # so hidden layers may include the current location without leaking.
        self.input_layer.set_mask(degrees, strict=True)
        for block in self.blocks:
            block.conv.set_mask(degrees, strict=False)
        self.input_degrees.copy_(degrees.to(self.input_degrees.device))
        self.mask_index.fill_(index)

    @torch.no_grad()
    def resample_masks(self) -> None:
        self.set_masks(int(self.mask_index.item()) + 1)

    def forward(self, inputs: Tensor) -> Tensor:
        image = inputs.reshape(-1, 1, self.side, self.side)
        hidden = _gated_activation(self.input_layer(image))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_head(hidden).flatten(1)

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
            for index in range(start_index, start_index + min(masks, self.ORDER_COUNT)):
                self.set_masks(index)
                log_probabilities.append(self.log_prob(inputs))
        finally:
            self.set_masks(original_index)
        stacked = torch.stack(log_probabilities, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(stacked.shape[0])

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
