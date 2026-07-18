"""Fixed MNIST classifier used solely as the KID feature extractor."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import MNIST

from iwae_reproduction.data import seeded_dataloader


class MNISTFeatureClassifier(L.LightningModule):
    """Small deterministic classifier exposing a 128-dimensional feature space."""

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_layer = nn.Linear(64 * 7 * 7, 128)
        self.output_layer = nn.Linear(128, 10)
        # Micro averaging preserves the original overall correct/total statistic.
        self.train_accuracy = MulticlassAccuracy(num_classes=10, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=10, average="micro")
        self.test_accuracy = MulticlassAccuracy(num_classes=10, average="micro")

    def features(self, images: Tensor) -> Tensor:
        hidden = self.convolutions(images).flatten(1)
        return torch.relu(self.feature_layer(hidden))

    def forward(self, images: Tensor) -> Tensor:
        return self.output_layer(self.features(images))

    def _shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        prefix: str,
        accuracy: MulticlassAccuracy,
    ) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        accuracy.update(logits, labels)
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            f"{prefix}/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train", self.train_accuracy)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, "val", self.val_accuracy)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, "test", self.test_accuracy)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])


class ClassifierDataModule(L.LightningDataModule):
    """Static MNIST loaders for the auxiliary feature classifier."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        batch_size: int = 256,
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
        self.train_data: TensorDataset | None = None
        self.val_data: TensorDataset | None = None
        self.test_data: TensorDataset | None = None

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    @staticmethod
    def _dataset(images: Tensor, labels: Tensor) -> TensorDataset:
        return TensorDataset(images.to(torch.float32).div(255).unsqueeze(1), labels)

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"} and self.train_data is None:
            train = MNIST(self.data_dir, train=True, download=False)
            self.train_data = self._dataset(train.data[:59_600], train.targets[:59_600])
            self.val_data = self._dataset(train.data[59_600:], train.targets[59_600:])
        if stage in {None, "test", "predict"} and self.test_data is None:
            test = MNIST(self.data_dir, train=False, download=False)
            self.test_data = self._dataset(test.data, test.targets)

    def _loader(self, dataset: TensorDataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("setup() must be called before requesting a loader")
        return seeded_dataloader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_data, shuffle=False)


class MNISTFeatureExtractor(nn.Module):
    """Expose classifier features in TorchMetrics' expected module interface."""

    def __init__(self, classifier: MNISTFeatureClassifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier.features(images)
