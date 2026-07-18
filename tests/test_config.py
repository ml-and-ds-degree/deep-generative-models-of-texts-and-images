from __future__ import annotations

import unittest

import torch

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
