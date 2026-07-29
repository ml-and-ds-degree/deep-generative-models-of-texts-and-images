from __future__ import annotations

import unittest

import torch

from iwae_reproduction.classifier import MNISTFeatureClassifier, MNISTFeatureExtractor


class KIDFeatureTests(unittest.TestCase):
    def test_feature_extractor_exposes_the_fixed_128_dimensional_space(self) -> None:
        classifier = MNISTFeatureClassifier().eval()
        extractor = MNISTFeatureExtractor(classifier)
        images = torch.rand(3, 1, 28, 28)

        self.assertEqual(extractor(images).shape, (3, 128))
        self.assertEqual(classifier(images).shape, (3, 10))
