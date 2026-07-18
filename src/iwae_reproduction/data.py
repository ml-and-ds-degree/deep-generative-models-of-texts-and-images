"""Dynamically binarized MNIST with the split used by the IWAE paper."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


class BinarizedImages(Dataset[Tensor]):
    """MNIST probabilities sampled as Bernoulli observations.

    ``seed=None`` implements the paper's dynamic training binarization. A seed
    fixes one draw per example for paired validation/test comparisons.
    """

    def __init__(self, images: Tensor, *, seed: int | None = None):
        self.probabilities = images.to(torch.float32).div_(255).flatten(1)
        self.seed = seed

    def __len__(self) -> int:
        return self.probabilities.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        probability = self.probabilities[index]
        if self.seed is None:
            return torch.bernoulli(probability)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + index)
        return torch.bernoulli(probability, generator=generator)


def seeded_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    """Build a reproducible loader shared by both MNIST data modules."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


class MNISTDataModule(L.LightningDataModule):
    """Paper-code split with dynamic train and fixed paired-evaluation draws.

    The paper describes the standard 60,000/10,000 train/test split; its
    released code reserves the final 400 training examples for validation.
    Fixed validation/test draws are this project's documented reproducibility
    addition, not a claim about the paper's evaluation protocol.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        batch_size: int = 20,
        num_workers: int = 0,
        seed: int = 236,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "data_dir": str(data_dir),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "seed": seed,
            }
        )
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_data: BinarizedImages | None = None
        self.val_data: BinarizedImages | None = None
        self.test_data: BinarizedImages | None = None
        self.pixel_mean: Tensor | None = None

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"} and self.train_data is None:
            train = MNIST(self.data_dir, train=True, download=False)
            train_images = train.data[:59_600]
            self.train_data = BinarizedImages(train_images)
            self.val_data = BinarizedImages(train.data[59_600:], seed=self.seed + 1_000_000)
            self.pixel_mean = train_images.to(torch.float32).mean(0).div(255).flatten()
        if stage in {None, "test", "predict"} and self.test_data is None:
            test = MNIST(self.data_dir, train=False, download=False)
            self.test_data = BinarizedImages(test.data, seed=self.seed + 2_000_000)

    def _loader(self, dataset: Dataset[Tensor], *, shuffle: bool) -> DataLoader[Tensor]:
        return seeded_dataloader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
            shuffle=shuffle,
        )

    def _require(self, value: BinarizedImages | None, name: str) -> BinarizedImages:
        if value is None:
            raise RuntimeError(f"setup() must be called before {name}()")
        return value

    def train_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.train_data, "train_dataloader"), shuffle=True)

    def val_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.val_data, "val_dataloader"), shuffle=False)

    def test_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.test_data, "test_dataloader"), shuffle=False)
