from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from made_reproduction.config import DATASETS, DatasetName, DatasetSpec, paper_preset
from made_reproduction.data import BinaryVectors, load_original_splits, sha256sum


class MADEDataTests(unittest.TestCase):
    def test_binarized_mnist_preset_matches_paper_condition(self) -> None:
        spec = DATASETS[DatasetName.BINARIZED_MNIST]
        preset = paper_preset(DatasetName.BINARIZED_MNIST)
        self.assertEqual((spec.input_dim, spec.train_size), (784, 50_000))
        self.assertEqual(preset.hidden_dims, (8_000,))
        self.assertEqual(preset.optimizer.value, "adagrad")
        self.assertFalse(preset.direct_input_to_output)
        self.assertEqual((preset.paper_test_nll, preset.paper_test_nll_ci95), (88.40, 0.45))

    def test_binary_vectors_reject_non_binary_values(self) -> None:
        with self.assertRaises(ValueError):
            BinaryVectors(np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32))

    def test_original_npz_schema_is_validated(self) -> None:
        spec = DatasetSpec(
            name=DatasetName.ADULT,
            input_dim=3,
            train_size=2,
            validation_size=1,
            test_size=1,
            url="https://example.invalid/data.npz",
            sha256="unused",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.npz"
            np.savez(
                path,
                inputsize=np.asarray(3),
                train_length=np.asarray(2),
                valid_length=np.asarray(1),
                test_length=np.asarray(1),
                train_data=np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.float32),
                valid_data=np.asarray([[0, 0, 1]], dtype=np.float32),
                test_data=np.asarray([[1, 1, 0]], dtype=np.float32),
            )
            splits = load_original_splits(path, spec)
            self.assertEqual(len(splits["train"]), 2)
            self.assertEqual(splits["test"][0].shape, (3,))
            self.assertEqual(len(sha256sum(path)), 64)
