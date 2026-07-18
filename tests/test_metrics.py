from __future__ import annotations

import unittest

import torch

from iwae_reproduction.metrics import ActiveUnits


class ActiveUnitsTests(unittest.TestCase):
    def test_counts_dimensions_above_variance_threshold(self) -> None:
        metric = ActiveUnits(latent_dim=3, threshold=1e-2)
        metric.update(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )
        torch.testing.assert_close(metric.compute(), torch.tensor(1.0))

    def test_batch_accumulation_matches_single_update(self) -> None:
        means = torch.randn(20, 4, generator=torch.Generator().manual_seed(7))
        batched = ActiveUnits(4)
        batched.update(means[:8])
        batched.update(means[8:])
        single = ActiveUnits(4)
        single.update(means)
        torch.testing.assert_close(batched.compute(), single.compute())
