"""PyTorch Lightning training module for the MADE paper reproduction."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import torch
from torch import Tensor
from torchmetrics.aggregation import MeanMetric

from made_reproduction.config import (
    ActivationName,
    ArchitectureName,
    MaskMode,
    OptimizerName,
)
from made_reproduction.networks import (
    MADE,
    LocallyMaskedConvMADE,
    PixelCNNMADE,
    ResidualMADE,
)


class MADELitModule(L.LightningModule):
    """Exact Bernoulli maximum-likelihood training for MADE."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int] = (500,),
        architecture: ArchitectureName | str = ArchitectureName.PAPER,
        residual_blocks: int = 0,
        hidden_to_output_skips: bool = False,
        zero_init_final_branch: bool = False,
        activation: ActivationName | str = ActivationName.TANH,
        direct_input_to_output: bool = True,
        mask_seed: int = 1234,
        mask_mode: MaskMode | str = MaskMode.FIXED,
        validation_masks: int = 1,
        test_masks: int = 1,
        optimizer: OptimizerName | str = OptimizerName.SGD,
        learning_rate: float = 0.01,
        decay: float = 0.0,
        epsilon: float = 0.0,
    ):
        super().__init__()
        self.mask_mode = MaskMode(mask_mode)
        self.optimizer_name = OptimizerName(optimizer)
        self.architecture = ArchitectureName(architecture)
        activation_name = ActivationName(activation)
        self.save_hyperparameters(
            {
                "input_dim": input_dim,
                "hidden_dims": tuple(hidden_dims),
                "architecture": self.architecture.value,
                "residual_blocks": residual_blocks,
                "hidden_to_output_skips": hidden_to_output_skips,
                "zero_init_final_branch": zero_init_final_branch,
                "activation": activation_name.value,
                "direct_input_to_output": direct_input_to_output,
                "mask_seed": mask_seed,
                "mask_mode": self.mask_mode.value,
                "validation_masks": validation_masks,
                "test_masks": test_masks,
                "optimizer": self.optimizer_name.value,
                "learning_rate": learning_rate,
                "decay": decay,
                "epsilon": epsilon,
            }
        )
        if validation_masks < 1 or test_masks < 1:
            raise ValueError("validation_masks and test_masks must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.architecture in (
            ArchitectureName.PAPER,
            ArchitectureName.DIRECT,
            ArchitectureName.DEEP,
        ):
            if residual_blocks:
                raise ValueError("non-residual MADE does not accept residual blocks")
            self.model = MADE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation=activation_name,
                direct_input_to_output=direct_input_to_output,
                hidden_to_output_skips=hidden_to_output_skips,
                zero_init_final_branch=zero_init_final_branch,
                mask_seed=mask_seed,
            )
        elif self.architecture is ArchitectureName.RESIDUAL:
            if len(hidden_dims) != 1:
                raise ValueError("residual MADE uses exactly one hidden width")
            self.model = ResidualMADE(
                input_dim=input_dim,
                hidden_dim=hidden_dims[0],
                residual_blocks=residual_blocks,
                activation=activation_name,
                direct_input_to_output=direct_input_to_output,
                mask_seed=mask_seed,
            )
        elif self.architecture is ArchitectureName.LMCONV:
            if len(hidden_dims) != 1:
                raise ValueError("LMConv MADE uses one channel width")
            self.model = LocallyMaskedConvMADE(
                input_dim=input_dim,
                channels=hidden_dims[0],
                residual_blocks=residual_blocks,
                mask_seed=mask_seed,
                direct_input_to_output=direct_input_to_output,
            )
        else:
            if len(hidden_dims) != 1:
                raise ValueError("PixelCNN MADE uses one channel width")
            self.model = PixelCNNMADE(
                input_dim=input_dim,
                channels=hidden_dims[0],
                residual_blocks=residual_blocks,
                direct_input_to_output=direct_input_to_output,
            )
        self.test_nll = MeanMetric()

    def _log_prob(self, batch: Tensor, *, masks: int, start_index: int) -> Tensor:
        if masks == 1 and self.mask_mode is MaskMode.FIXED:
            return self.model.log_prob(batch)
        return self.model.ensemble_log_prob(batch, masks, start_index=start_index)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        del batch_idx
        nll = -self.model.log_prob(batch).mean()
        self.log(
            "train/nll",
            nll,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.shape[0],
        )
        return nll

    def on_train_batch_end(
        self,
        outputs: Tensor | dict[str, Tensor] | None,
        batch: Tensor,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        # Automatic optimization has completed its backward pass and optimizer
        # step before this hook. Changing a mask earlier would mutate a tensor
        # retained by autograd for the current step.
        if self.mask_mode is MaskMode.EVERY_BATCH:
            self.model.resample_masks()

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        del batch_idx
        log_prob = self._log_prob(
            batch,
            masks=self.hparams.validation_masks,
            start_index=1_000_000,
        )
        self.log(
            "val/nll",
            -log_prob.mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.shape[0],
        )

    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        del batch_idx
        log_prob = self._log_prob(
            batch,
            masks=self.hparams.test_masks,
            start_index=2_000_000,
        )
        self.test_nll.update(-log_prob)

    def on_test_epoch_end(self) -> None:
        self.log("test/nll", self.test_nll)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer_name is OptimizerName.SGD:
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        if self.optimizer_name is OptimizerName.ADAGRAD:
            return torch.optim.Adagrad(
                self.parameters(),
                lr=self.hparams.learning_rate,
                eps=self.hparams.epsilon,
            )
        return torch.optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            rho=self.hparams.decay,
            eps=self.hparams.epsilon,
        )
