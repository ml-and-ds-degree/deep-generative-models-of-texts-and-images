"""Checksum-pinned binary datasets from the authors' MADE repository."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from made_reproduction.config import DATASETS, DatasetName, DatasetSpec


def sha256sum(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest without loading an entire dataset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(spec: DatasetSpec, destination: Path) -> None:
    """Download an original dataset atomically and verify its pinned digest."""

    if destination.exists():
        observed = sha256sum(destination)
        if observed != spec.sha256:
            raise ValueError(
                f"checksum mismatch for {destination}: expected {spec.sha256}, got {observed}"
            )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "made-reproduction"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        observed = sha256sum(partial)
        if observed != spec.sha256:
            raise ValueError(
                f"downloaded checksum mismatch: expected {spec.sha256}, got {observed}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


class BinaryVectors(Dataset[Tensor]):
    """Tensor-backed binary observations returned without labels or transforms."""

    def __init__(self, values: np.ndarray | Tensor):
        self.values = torch.as_tensor(values, dtype=torch.float32).contiguous()
        if self.values.ndim != 2:
            raise ValueError("binary observations must have shape [examples, dimensions]")
        if not torch.all((self.values == 0) | (self.values == 1)):
            raise ValueError("MADE's Bernoulli likelihood requires binary observations")

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.values[index]


def load_original_splits(path: Path, spec: DatasetSpec) -> dict[str, BinaryVectors]:
    """Load and validate the exact train/validation/test arrays in an original NPZ."""

    # NumPy warns that these Python-2-era archives require additional header
    # parsing. Preserving the authors' bytes is more important than rewriting
    # the files merely to remove that harmless warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Reading `\.npy` or `\.npz` file required additional header parsing.*",
            category=UserWarning,
        )
        with np.load(path, allow_pickle=False) as archive:
            input_dim = int(archive["inputsize"])
            arrays = {
                "train": np.asarray(archive["train_data"], dtype=np.float32),
                "validation": np.asarray(archive["valid_data"], dtype=np.float32),
                "test": np.asarray(archive["test_data"], dtype=np.float32),
            }
            recorded_sizes = {
                "train": int(archive["train_length"]),
                "validation": int(archive["valid_length"]),
                "test": int(archive["test_length"]),
            }

    expected_sizes = {
        "train": spec.train_size,
        "validation": spec.validation_size,
        "test": spec.test_size,
    }
    if input_dim != spec.input_dim:
        raise ValueError(f"expected input dimension {spec.input_dim}, found {input_dim}")
    for split, values in arrays.items():
        expected_shape = (expected_sizes[split], spec.input_dim)
        if values.shape != expected_shape or recorded_sizes[split] != expected_sizes[split]:
            raise ValueError(
                f"invalid {split} split: expected {expected_shape}, found {values.shape}"
            )
    return {split: BinaryVectors(values) for split, values in arrays.items()}


class MADEDataModule(L.LightningDataModule):
    """Paper splits and fixed-order training for the binary MADE benchmarks."""

    def __init__(
        self,
        dataset: DatasetName | str = DatasetName.BINARIZED_MNIST,
        data_dir: str | Path = "data",
        batch_size: int = 100,
        num_workers: int = 0,
        seed: int = 1234,
    ):
        super().__init__()
        self.dataset_name = DatasetName(dataset)
        self.spec = DATASETS[self.dataset_name]
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_data: BinaryVectors | None = None
        self.validation_data: BinaryVectors | None = None
        self.test_data: BinaryVectors | None = None
        self.save_hyperparameters(
            {
                "dataset": self.dataset_name.value,
                "data_dir": str(self.data_dir),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "seed": seed,
            }
        )

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "made" / f"{self.dataset_name.value}.npz"

    def prepare_data(self) -> None:
        download_dataset(self.spec, self.dataset_path)

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_data is not None:
            return
        splits = load_original_splits(self.dataset_path, self.spec)
        self.train_data = splits["train"]
        self.validation_data = splits["validation"]
        self.test_data = splits["test"]

    def _require(self, value: BinaryVectors | None, name: str) -> BinaryVectors:
        if value is None:
            raise RuntimeError(f"setup() must be called before {name}()")
        return value

    def _loader(self, dataset: BinaryVectors) -> DataLoader[Tensor]:
        # The released Theano experiment traverses the already prepared array
        # in order. Keeping shuffle=False is therefore paper fidelity, and it
        # also makes checkpoint resumption independent of sampler RNG state.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.train_data, "train_dataloader"))

    def val_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.validation_data, "val_dataloader"))

    def test_dataloader(self) -> DataLoader[Tensor]:
        return self._loader(self._require(self.test_data, "test_dataloader"))
