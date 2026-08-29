from __future__ import annotations

import unittest

import lightning as L
import torch
from torch.utils.data import DataLoader

from made_reproduction.metrics import unbiased_rbf_mmd
from made_reproduction.module import MADELitModule
from made_reproduction.networks import (
    MADE,
    AttentionMADE,
    GatedPixelCNN,
    LocallyMaskedConvMADE,
    OrderEnsembleLMConvMADE,
    PixelCNNMADE,
    ResidualMADE,
    s_curve_order_degrees,
)


class MADEArchitectureTests(unittest.TestCase):
    def test_output_shape_and_finite_log_probability(self) -> None:
        model = MADE(8, (12, 10), mask_seed=7)
        inputs = torch.bernoulli(torch.full((4, 8), 0.5))
        self.assertEqual(model(inputs).shape, (4, 8))
        self.assertEqual(model.log_prob(inputs).shape, (4,))
        self.assertTrue(torch.isfinite(model.log_prob(inputs)).all())

    def test_forbidden_future_dependencies_have_zero_jacobian(self) -> None:
        model = MADE(7, (11,), mask_seed=13, direct_input_to_output=True)
        inputs = torch.randn(1, 7, requires_grad=True)
        logits = model(inputs)
        for output_index in range(7):
            gradient = torch.autograd.grad(
                logits[0, output_index],
                inputs,
                retain_graph=True,
            )[0][0]
            forbidden = model.input_degrees >= model.input_degrees[output_index]
            torch.testing.assert_close(gradient[forbidden], torch.zeros_like(gradient[forbidden]))

    def test_mask_indices_are_reproducible(self) -> None:
        first = MADE(9, (15,), mask_seed=23)
        second = MADE(9, (15,), mask_seed=23)
        torch.testing.assert_close(first.input_degrees, second.input_degrees)
        torch.testing.assert_close(first.layers[0].mask, second.layers[0].mask)
        old_mask = first.layers[0].mask.clone()
        first.resample_masks()
        self.assertFalse(torch.equal(old_mask, first.layers[0].mask))

    def test_ancestral_samples_are_binary(self) -> None:
        model = MADE(6, (8,), mask_seed=5)
        generator = torch.Generator().manual_seed(19)
        samples = model.sample(10, generator=generator)
        self.assertEqual(samples.shape, (10, 6))
        self.assertTrue(torch.all((samples == 0) | (samples == 1)))

    def test_residual_made_preserves_autoregressive_dependencies(self) -> None:
        model = ResidualMADE(7, 12, 2, mask_seed=17, direct_input_to_output=True)
        inputs = torch.randn(1, 7, requires_grad=True)
        logits = model(inputs)
        for output_index in range(7):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            forbidden = model.input_degrees >= model.input_degrees[output_index]
            torch.testing.assert_close(gradient[forbidden], torch.zeros_like(gradient[forbidden]))

    def test_bottleneck_skip_preserves_autoregressive_dependencies(self) -> None:
        model = MADE(
            7,
            (12, 4),
            mask_seed=29,
            direct_input_to_output=True,
            hidden_to_output_skips=True,
            zero_init_final_branch=True,
        )
        inputs = torch.randn(1, 7, requires_grad=True)
        logits = model(inputs)
        for output_index in range(7):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            forbidden = model.input_degrees >= model.input_degrees[output_index]
            torch.testing.assert_close(gradient[forbidden], torch.zeros_like(gradient[forbidden]))

    def test_lmconv_preserves_arbitrary_order_dependencies(self) -> None:
        model = LocallyMaskedConvMADE(
            16, channels=4, residual_blocks=2, mask_seed=31
        )
        inputs = torch.randn(1, 16, requires_grad=True)
        logits = model(inputs)
        for output_index in range(16):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            forbidden = model.input_degrees >= model.input_degrees[output_index]
            torch.testing.assert_close(gradient[forbidden], torch.zeros_like(gradient[forbidden]))

    def test_pixelcnn_preserves_raster_order_dependencies(self) -> None:
        model = PixelCNNMADE(16, channels=4, residual_blocks=2)
        inputs = torch.randn(1, 16, requires_grad=True)
        logits = model(inputs)
        for output_index in range(16):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            torch.testing.assert_close(
                gradient[output_index:], torch.zeros_like(gradient[output_index:])
            )

    def test_gated_pixelcnn_preserves_raster_order_dependencies(self) -> None:
        model = GatedPixelCNN(16, channels=4, residual_blocks=2)
        # The output head is zero-initialized for training stability, which
        # would make this test vacuous; randomize it so gradients flow.
        for layer in model.output_head:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)
        inputs = torch.randn(1, 16, requires_grad=True)
        logits = model(inputs)
        for output_index in range(16):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            torch.testing.assert_close(
                gradient[output_index:], torch.zeros_like(gradient[output_index:])
            )

    def test_gated_pixelcnn_uses_past_pixels(self) -> None:
        model = GatedPixelCNN(16, channels=4, residual_blocks=2)
        for layer in model.output_head:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)
        inputs = torch.randn(1, 16, requires_grad=True)
        gradient = torch.autograd.grad(model(inputs)[0, 15], inputs)[0][0]
        self.assertGreater(gradient[:15].abs().sum().item(), 0.0)

    def test_s_curve_orders_are_distinct_permutations(self) -> None:
        orders = [s_curve_order_degrees(4, index) for index in range(8)]
        for degrees in orders:
            self.assertEqual(sorted(degrees.tolist()), list(range(1, 17)))
        as_tuples = {tuple(degrees.tolist()) for degrees in orders}
        self.assertEqual(len(as_tuples), 8)

    def test_lmconv_ensemble_preserves_dependencies_in_every_order(self) -> None:
        model = OrderEnsembleLMConvMADE(16, channels=4, residual_blocks=2)
        for layer in model.output_head:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)
        for order_index in range(8):
            model.set_masks(order_index)
            inputs = torch.randn(1, 16, requires_grad=True)
            logits = model(inputs)
            for output_index in range(16):
                gradient = torch.autograd.grad(
                    logits[0, output_index], inputs, retain_graph=True
                )[0][0]
                forbidden = model.input_degrees >= model.input_degrees[output_index]
                torch.testing.assert_close(
                    gradient[forbidden], torch.zeros_like(gradient[forbidden])
                )

    def test_lmconv_ensemble_log_prob_averages_orders(self) -> None:
        model = OrderEnsembleLMConvMADE(16, channels=4, residual_blocks=1)
        inputs = torch.bernoulli(torch.full((3, 16), 0.5))
        ensemble = model.ensemble_log_prob(inputs, 8, start_index=2_000_000)
        self.assertEqual(ensemble.shape, (3,))
        self.assertTrue(torch.isfinite(ensemble).all())
        self.assertEqual(int(model.mask_index.item()), 0)

    def test_attention_made_preserves_autoregressive_dependencies(self) -> None:
        model = AttentionMADE(
            8, 16, 2, num_heads=4, dropout=0.0, mask_seed=41
        ).eval()
        inputs = torch.randn(1, 8, requires_grad=True)
        logits = model(inputs)
        for output_index in range(8):
            gradient = torch.autograd.grad(
                logits[0, output_index], inputs, retain_graph=True
            )[0][0]
            forbidden = model.input_degrees >= model.input_degrees[output_index]
            torch.testing.assert_close(
                gradient[forbidden],
                torch.zeros_like(gradient[forbidden]),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_attention_made_mask_indices_are_reproducible(self) -> None:
        first = AttentionMADE(9, 16, 1, num_heads=4, dropout=0.0, mask_seed=23)
        second = AttentionMADE(9, 16, 1, num_heads=4, dropout=0.0, mask_seed=23)
        torch.testing.assert_close(first.input_degrees, second.input_degrees)
        torch.testing.assert_close(first.generation_order, second.generation_order)
        old_order = first.generation_order.clone()
        first.resample_masks()
        self.assertFalse(torch.equal(old_order, first.generation_order))

    def test_unbiased_mmd_is_permutation_invariant(self) -> None:
        values = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        permuted = values[torch.tensor([2, 0, 1])]
        forward = unbiased_rbf_mmd(values, permuted, bandwidth_squared=1.0)
        backward = unbiased_rbf_mmd(permuted, values, bandwidth_squared=1.0)
        torch.testing.assert_close(forward, backward)


class MADELightningTests(unittest.TestCase):
    def test_adagrad_uses_paper_epsilon(self) -> None:
        model = MADELitModule(
            input_dim=8,
            hidden_dims=(10,),
            optimizer="adagrad",
            learning_rate=0.01,
            epsilon=1e-6,
        )
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adagrad)
        self.assertEqual(optimizer.defaults["eps"], 1e-6)

    def test_both_mask_modes_complete_a_training_step(self) -> None:
        L.seed_everything(11)
        batches = DataLoader(torch.bernoulli(torch.full((6, 8), 0.5)), batch_size=3)
        for mask_mode in ("fixed", "every-batch"):
            model = MADELitModule(
                input_dim=8,
                hidden_dims=(10,),
                mask_seed=11,
                mask_mode=mask_mode,
                validation_masks=2,
            )
            trainer = L.Trainer(
                accelerator="cpu",
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=True,
            )
            trainer.fit(model, train_dataloaders=batches, val_dataloaders=batches)

    def test_improved_conv_architectures_complete_a_training_step(self) -> None:
        L.seed_everything(11)
        batches = DataLoader(torch.bernoulli(torch.full((6, 16), 0.5)), batch_size=3)
        cases = (
            ("gated-pixelcnn", "fixed", 1, 1),
            ("lmconv-ensemble", "every-batch", 8, 8),
        )
        for architecture, mask_mode, validation_masks, test_masks in cases:
            model = MADELitModule(
                input_dim=16,
                hidden_dims=(8,),
                architecture=architecture,
                residual_blocks=2,
                direct_input_to_output=False,
                mask_mode=mask_mode,
                validation_masks=validation_masks,
                test_masks=test_masks,
                optimizer="adamw",
                learning_rate=1e-3,
                mask_seed=11,
            )
            trainer = L.Trainer(
                accelerator="cpu",
                max_epochs=1,
                limit_train_batches=2,
                limit_val_batches=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=True,
            )
            trainer.fit(model, train_dataloaders=batches, val_dataloaders=batches)

    def test_attention_architecture_completes_a_training_step(self) -> None:
        L.seed_everything(11)
        batches = DataLoader(torch.bernoulli(torch.full((6, 8), 0.5)), batch_size=3)
        model = MADELitModule(
            input_dim=8,
            hidden_dims=(16,),
            architecture="attention",
            residual_blocks=2,
            num_heads=4,
            dropout=0.0,
            optimizer="adamw",
            learning_rate=1e-3,
            mask_seed=11,
        )
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=batches, val_dataloaders=batches)
