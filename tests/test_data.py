from __future__ import annotations

import unittest

import torch

from iwae_reproduction.data import BinarizedImages


class BinarizedImagesTests(unittest.TestCase):
    def test_fixed_draw_is_stable_per_example(self) -> None:
        images = torch.full((2, 28, 28), 127, dtype=torch.uint8)
        dataset = BinarizedImages(images, seed=17)
        self.assertTrue(torch.equal(dataset[0], dataset[0]))
        self.assertFalse(torch.equal(dataset[0], dataset[1]))

    def test_dynamic_draw_changes(self) -> None:
        torch.manual_seed(1)
        images = torch.full((1, 28, 28), 127, dtype=torch.uint8)
        dataset = BinarizedImages(images)
        self.assertFalse(torch.equal(dataset[0], dataset[0]))
