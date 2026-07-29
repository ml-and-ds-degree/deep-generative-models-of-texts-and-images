from __future__ import annotations

import unittest

import torch

from iwae_reproduction.cli import _progressive_particle_schedule
from iwae_reproduction.config import PAPER_EPOCH_BOUNDARIES, PAPER_LR_DECAY


class PaperScheduleTests(unittest.TestCase):
    def test_multistep_scheduler_matches_paper_boundaries(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=PAPER_EPOCH_BOUNDARIES[:-1],
            gamma=PAPER_LR_DECAY,
        )
        observed = []
        for _epoch in range(5):
            observed.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(observed[0], 1e-3)
        self.assertAlmostEqual(observed[1], 1e-3 * 10 ** (-1 / 7))
        self.assertAlmostEqual(observed[3], 1e-3 * 10 ** (-1 / 7))
        self.assertAlmostEqual(observed[4], 1e-3 * 10 ** (-2 / 7))


class ProgressiveScheduleTests(unittest.TestCase):
    def test_default_fast_schedule_reserves_final_sixth_for_full_particles(self) -> None:
        counts, boundaries = _progressive_particle_schedule(121, 5, 10, 50)
        self.assertEqual(counts, (5, 10, 50))
        self.assertEqual(boundaries, (60, 100))

    def test_short_run_falls_back_to_final_particle_count(self) -> None:
        self.assertEqual(_progressive_particle_schedule(5, 5, 10, 50), ((50,), ()))

    def test_particle_schedule_must_be_non_decreasing(self) -> None:
        with self.assertRaises(ValueError):
            _progressive_particle_schedule(121, 10, 5, 50)
