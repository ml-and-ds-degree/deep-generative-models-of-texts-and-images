"""PyTorch Lightning implementation of IWAE and IWAE-DReG."""

from __future__ import annotations

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.aggregation import MeanMetric

from iwae_reproduction.config import PAPER_EPOCH_BOUNDARIES, PAPER_LR_DECAY, Objective
from iwae_reproduction.metrics import ActiveUnits
from iwae_reproduction.networks import (
    BernoulliDecoder,
    GaussianEncoder,
    sample_diagonal_gaussian,
)
from iwae_reproduction.objectives import (
    dreg_encoder_loss,
    importance_terms,
    iwae_loss,
)


class IWAELitModule(L.LightningModule):
    """One-stochastic-layer IWAE with the original or DReG encoder estimator."""

    def __init__(
        self,
        *,
        input_dim: int = 784,
        hidden_dim: int = 200,
        latent_dim: int = 50,
        train_particles: int = 50,
        objective: str = "iwae",
        learning_rate: float = 1e-3,
        validation_particles: int = 50,
        test_particles: int = 5_000,
        evaluation_chunk_size: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.objective = Objective(objective)
        self.encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, hidden_dim, input_dim)
        self.test_nll = MeanMetric()
        self.test_active_units = ActiveUnits(latent_dim)

    @torch.no_grad()
    def initialize_decoder_bias(self, pixel_mean: Tensor) -> None:
        self.decoder.initialize_output_bias(pixel_mean)

    def _draw_terms(self, x: Tensor, particles: int) -> tuple[Tensor, Tensor]:
        mean, log_std = self.encoder(x)
        z = sample_diagonal_gaussian(mean, log_std, particles)
        logits = self.decoder(z)
        terms = importance_terms(x, z, mean, log_std, logits)
        return terms.log_weights, terms.pathwise_log_weights

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        del batch_idx
        encoder_optimizer, decoder_optimizer = self.optimizers()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        log_weights, pathwise_log_weights = self._draw_terms(batch, self.hparams.train_particles)
        bound_loss = iwae_loss(log_weights)
        if self.objective is Objective.IWAE:
            self.manual_backward(bound_loss)
        else:
            # DReG is the scientific improvement: the decoder retains the
            # original IWAE gradient while the encoder uses its pathwise surrogate.
            encoder_parameters = tuple(self.encoder.parameters())
            decoder_parameters = tuple(self.decoder.parameters())
            with self.toggled_optimizer(decoder_optimizer):
                self.manual_backward(
                    bound_loss,
                    inputs=decoder_parameters,
                    retain_graph=True,
                )
            encoder_loss = dreg_encoder_loss(log_weights, pathwise_log_weights)
            with self.toggled_optimizer(encoder_optimizer):
                self.manual_backward(encoder_loss, inputs=encoder_parameters)
            self.log(
                "train/dreg_surrogate",
                encoder_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch.shape[0],
            )

        encoder_optimizer.step()
        decoder_optimizer.step()
        self.log(
            "train/nll_bound",
            bound_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.shape[0],
        )
        return bound_loss.detach()

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        del batch_idx
        log_weights, _ = self._draw_terms(batch, self.hparams.validation_particles)
        self.log(
            "val/nll_bound",
            iwae_loss(log_weights),
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.shape[0],
        )

    def estimate_log_likelihood(self, x: Tensor, particles: int, chunk_size: int) -> Tensor:
        """Estimate log p(x), retaining at most ``chunk_size`` particles.

        Chunking is an exact computational reformulation of the single
        L_particles log-sum-exp estimator; it bounds memory without averaging
        independent chunk estimates. The paper comparison uses L_5000.
        """
        if particles < 1 or chunk_size < 1:
            raise ValueError("particles and chunk_size must both be positive")
        mean, log_std = self.encoder(x)
        accumulated: Tensor | None = None
        drawn = 0
        while drawn < particles:
            current = min(chunk_size, particles - drawn)
            z = sample_diagonal_gaussian(mean, log_std, current)
            logits = self.decoder(z)
            log_weights = importance_terms(x, z, mean, log_std, logits).log_weights
            chunk_log_sum = torch.logsumexp(log_weights, dim=1)
            accumulated = (
                chunk_log_sum
                if accumulated is None
                else torch.logaddexp(accumulated, chunk_log_sum)
            )
            drawn += current
        if accumulated is None:
            raise RuntimeError("likelihood accumulation unexpectedly remained empty")
        return accumulated - torch.log(
            torch.tensor(float(particles), device=x.device, dtype=x.dtype)
        )

    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        del batch_idx
        log_likelihood = self.estimate_log_likelihood(
            batch,
            particles=self.hparams.test_particles,
            chunk_size=self.hparams.evaluation_chunk_size,
        )
        mean, _ = self.encoder(batch)
        self.test_nll.update(-log_likelihood)
        self.test_active_units.update(mean)

    def on_test_epoch_end(self) -> None:
        self.log(f"test/nll_l{self.hparams.test_particles}", self.test_nll)
        self.log("test/active_units", self.test_active_units)

    @torch.no_grad()
    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        z = torch.randn(
            count,
            self.hparams.latent_dim,
            device=self.device,
            generator=generator,
        )
        probabilities = torch.sigmoid(self.decoder(z))
        return probabilities.reshape(count, 1, 28, 28)

    def on_train_epoch_start(self) -> None:
        learning_rate = self.optimizers()[0].param_groups[0]["lr"]
        self.log("train/learning_rate", learning_rate, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Adam], list[MultiStepLR]]:
        adam_arguments = {
            "lr": self.hparams.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-4,
        }
        # Separate optimizers let DReG apply different encoder and decoder
        # gradients. Adam state is parameter-local, and both optimizers step
        # exactly once per batch, so this does not change the IWAE control.
        optimizers = [
            torch.optim.Adam(self.encoder.parameters(), **adam_arguments),
            torch.optim.Adam(self.decoder.parameters(), **adam_arguments),
        ]
        schedulers = [
            # Framework-native expression of the paper schedule; this is an
            # engineering modernization, not an experimental modification.
            MultiStepLR(
                optimizer,
                milestones=PAPER_EPOCH_BOUNDARIES[:-1],
                gamma=PAPER_LR_DECAY,
            )
            for optimizer in optimizers
        ]
        return optimizers, schedulers
