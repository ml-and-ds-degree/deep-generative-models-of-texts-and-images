from __future__ import annotations

import unittest

import lightning as L
import torch
from torch.utils.data import DataLoader

from iwae_reproduction.module import IWAELitModule
from iwae_reproduction.networks import BernoulliDecoder, GaussianEncoder
from iwae_reproduction.objectives import dreg_encoder_loss, iwae_loss


class ArchitectureTests(unittest.TestCase):
    def test_paper_architecture_shapes(self) -> None:
        encoder = GaussianEncoder()
        decoder = BernoulliDecoder()
        images = torch.rand(7, 784)
        mean, log_std = encoder(images)
        self.assertEqual(mean.shape, (7, 50))
        self.assertEqual(log_std.shape, (7, 50))
        self.assertEqual(decoder(torch.rand(7, 11, 50)).shape, (7, 11, 784))

    def test_dreg_surrogate_treats_normalized_weights_as_constants(self) -> None:
        log_weights = torch.randn(4, 5, requires_grad=True)
        pathwise = torch.randn(4, 5, requires_grad=True)
        loss = dreg_encoder_loss(log_weights, pathwise)
        gradient_log_weights, gradient_pathwise = torch.autograd.grad(
            loss, (log_weights, pathwise), allow_unused=True
        )
        self.assertIsNone(gradient_log_weights)
        self.assertIsNotNone(gradient_pathwise)

    def test_iwae_k1_is_negative_mean_log_weight(self) -> None:
        log_weights = torch.tensor([[1.0], [3.0]])
        torch.testing.assert_close(iwae_loss(log_weights), torch.tensor(-2.0))

    def test_progressive_particle_schedule_uses_epoch_boundaries(self) -> None:
        model = IWAELitModule(
            train_particles=50,
            training_particle_counts=(5, 10, 50),
            training_particle_boundaries=(3, 7),
        )
        observed = [model.training_particles_for_epoch(epoch) for epoch in (0, 2, 3, 6, 7)]
        self.assertEqual(observed, [5, 5, 10, 10, 50])

    def test_progressive_particle_schedule_requires_final_count(self) -> None:
        with self.assertRaises(ValueError):
            IWAELitModule(
                train_particles=50,
                training_particle_counts=(5, 10),
                training_particle_boundaries=(3,),
            )

    def test_progressive_particle_schedule_cannot_decrease(self) -> None:
        with self.assertRaises(ValueError):
            IWAELitModule(
                train_particles=50,
                training_particle_counts=(10, 5, 50),
                training_particle_boundaries=(3, 7),
            )


class LightningSmokeTests(unittest.TestCase):
    def test_both_estimators_complete_a_training_step(self) -> None:
        L.seed_everything(11)
        batches = DataLoader(torch.bernoulli(torch.full((4, 16), 0.5)), batch_size=2)
        for objective in ("iwae", "dreg"):
            model = IWAELitModule(
                input_dim=16,
                hidden_dim=8,
                latent_dim=3,
                train_particles=3,
                validation_particles=2,
                objective=objective,
            )
            trainer = L.Trainer(
                accelerator="cpu",
                max_epochs=1,
                limit_train_batches=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=True,
            )
            trainer.fit(model, train_dataloaders=batches)
