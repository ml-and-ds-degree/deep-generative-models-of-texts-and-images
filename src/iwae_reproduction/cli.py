"""Cyclopts command-line interface for every reproducible experiment."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.utils.benchmark as benchmark_utils
from cyclopts import App
from lightning import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import save_image

from iwae_reproduction.classifier import (
    ClassifierDataModule,
    MNISTFeatureClassifier,
    MNISTFeatureExtractor,
)
from iwae_reproduction.config import PAPER_EPOCH_BOUNDARIES, Objective
from iwae_reproduction.data import MNISTDataModule
from iwae_reproduction.module import IWAELitModule
from iwae_reproduction.objectives import iwae_loss

app = App(
    name="iwae",
    help="Reproduce one-layer IWAE and compare the DReG encoder estimator.",
)

Accelerator = Literal["auto", "cpu", "mps", "gpu", "tpu"]


def _fabric(accelerator: Accelerator) -> Fabric:
    fabric = Fabric(accelerator=accelerator, devices=1, precision="32-true")
    fabric.launch()
    return fabric


def _trainer(
    *,
    output_dir: Path,
    accelerator: Accelerator,
    max_epochs: int,
    deterministic: bool,
    callbacks: list | None = None,
    fast_dev_run: bool = False,
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
) -> L.Trainer:
    arguments: dict = {
        "accelerator": accelerator,
        "devices": 1,
        "max_epochs": max_epochs,
        "precision": "32-true",
        "deterministic": deterministic,
        "default_root_dir": output_dir,
        "callbacks": callbacks,
        "fast_dev_run": fast_dev_run,
    }
    if limit_train_batches is not None:
        arguments["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        arguments["limit_val_batches"] = limit_val_batches
    return L.Trainer(**arguments)


@app.command
def train(
    objective: Objective = Objective.IWAE,
    *,
    train_particles: int = 50,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs"),
    batch_size: int = 20,
    num_workers: int = 0,
    seed: int = 236,
    max_epochs: int = PAPER_EPOCH_BOUNDARIES[-1],
    accelerator: Accelerator = "auto",
    resume_from: Path | None = None,
    fast_dev_run: bool = False,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> None:
    """Train the paper reconstruction or its DReG improvement."""
    if train_particles < 1:
        raise ValueError("train_particles must be positive")
    L.seed_everything(seed, workers=True)
    data = MNISTDataModule(data_dir, batch_size, num_workers, seed)
    # The paper initializes the decoder from the empirical pixel mean, so the
    # DataModule is intentionally prepared before model construction.
    data.prepare_data()
    data.setup("fit")
    if resume_from is None:
        model = IWAELitModule(
            train_particles=train_particles,
            objective=objective.value,
            validation_particles=train_particles,
        )
        if data.pixel_mean is None:
            raise RuntimeError("MNIST pixel mean is unavailable")
        model.initialize_decoder_bias(data.pixel_mean)
    else:
        model = IWAELitModule.load_from_checkpoint(resume_from)
        if model.objective is not objective:
            raise ValueError(
                f"checkpoint objective is {model.objective.value!r}, not {objective.value!r}"
            )
        if model.hparams.train_particles != train_particles:
            raise ValueError(
                "checkpoint particle count does not match --train-particles: "
                f"{model.hparams.train_particles} != {train_particles}"
            )

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / objective.value / "checkpoints",
        filename="epoch-{epoch:04d}",
        auto_insert_metric_name=False,
        monitor="val/nll_bound",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    trainer = _trainer(
        output_dir=output_dir / objective.value,
        accelerator=accelerator,
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=[checkpoint],
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume_from)


@app.command
def evaluate(
    checkpoint: Path,
    *,
    data_dir: Path = Path("data"),
    batch_size: int = 20,
    num_workers: int = 0,
    seed: int = 236,
    particles: int = 5_000,
    chunk_size: int = 100,
    accelerator: Accelerator = "auto",
    limit_test_batches: int | None = None,
) -> None:
    """Report test NLL estimated with L_K and the number of active units."""
    L.seed_everything(seed, workers=True)
    data = MNISTDataModule(data_dir, batch_size, num_workers, seed)
    model = IWAELitModule.load_from_checkpoint(
        checkpoint,
        test_particles=particles,
        evaluation_chunk_size=chunk_size,
    )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        deterministic=True,
        logger=False,
        limit_test_batches=limit_test_batches,
    )
    trainer.test(model, datamodule=data)


@app.command
def sample(
    checkpoint: Path,
    *,
    output: Path = Path("samples.png"),
    count: int = 64,
    columns: int = 8,
    seed: int = 236,
    accelerator: Accelerator = "auto",
) -> None:
    """Save a deterministic grid of prior samples."""
    L.seed_everything(seed, workers=True)
    fabric = _fabric(accelerator)
    model = IWAELitModule.load_from_checkpoint(checkpoint).eval()
    model = fabric.setup_module(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        save_image(model.sample(count).cpu(), output, nrow=columns)
    print(output.resolve())


@app.command(name="train-classifier")
def train_classifier(
    *,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs/classifier"),
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 236,
    max_epochs: int = 5,
    accelerator: Accelerator = "auto",
    fast_dev_run: bool = False,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> None:
    """Train and test the frozen feature extractor used for MNIST KID."""
    L.seed_everything(seed, workers=True)
    data = ClassifierDataModule(data_dir, batch_size, num_workers, seed)
    model = MNISTFeatureClassifier()
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch-{epoch:02d}",
        auto_insert_metric_name=False,
        monitor="val/accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = _trainer(
        output_dir=output_dir,
        accelerator=accelerator,
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=[checkpoint],
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")


@app.command
def kid(
    checkpoint: Path,
    classifier_checkpoint: Path,
    *,
    data_dir: Path = Path("data"),
    count: int = 10_000,
    batch_size: int = 256,
    subset_size: int = 1_000,
    subsets: int = 50,
    seed: int = 236,
    accelerator: Accelerator = "auto",
) -> None:
    """Compute MNIST-classifier-feature KID for generated versus test images."""
    L.seed_everything(seed, workers=True)
    fabric = _fabric(accelerator)
    model = fabric.setup_module(IWAELitModule.load_from_checkpoint(checkpoint).eval())
    classifier = MNISTFeatureClassifier.load_from_checkpoint(classifier_checkpoint).eval()
    metric = KernelInceptionDistance(
        feature=MNISTFeatureExtractor(classifier),
        subsets=subsets,
        subset_size=subset_size,
    )
    metric = fabric.setup_module(metric)
    data = MNISTDataModule(data_dir, batch_size=batch_size, seed=seed)
    data.prepare_data()
    data.setup("test")
    loader = fabric.setup_dataloaders(data.test_dataloader())
    seen = 0
    with torch.inference_mode():
        for real in loader:
            remaining = count - seen
            if remaining <= 0:
                break
            real = real[:remaining]
            current = real.shape[0]
            generated = torch.bernoulli(model.sample(current))
            metric.update(real.reshape(current, 1, 28, 28), real=True)
            metric.update(generated, real=False)
            seen += current
    if seen < count:
        raise ValueError(f"requested {count} samples but only {seen} real examples are available")
    mean, standard_deviation = metric.compute()
    print(json.dumps({"kid_mean": mean.item(), "kid_std": standard_deviation.item()}))


@app.command
def benchmark(
    *,
    particles: int = 50,
    batch_size: int = 20,
    min_run_time: float = 1.0,
    seed: int = 236,
    accelerator: Accelerator = "auto",
) -> None:
    """Measure a representative forward/backward step on the selected device."""
    if particles < 1 or batch_size < 1 or min_run_time <= 0:
        raise ValueError("particles, batch_size, and min_run_time must be positive")
    L.seed_everything(seed, workers=True)
    fabric = _fabric(accelerator)
    model = IWAELitModule(train_particles=particles)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
    model, optimizer = fabric.setup(model, optimizer)
    images = torch.bernoulli(torch.full((batch_size, 784), 0.15, device=fabric.device))

    def benchmark_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        log_weights, _ = model._draw_terms(images, particles)
        loss = iwae_loss(log_weights)
        fabric.backward(loss)
        optimizer.step()

    timer = benchmark_utils.Timer(
        stmt="benchmark_step()",
        globals={"benchmark_step": benchmark_step},
        label="IWAE training step",
        sub_label=f"K={particles}, batch={batch_size}",
        description=str(fabric.device),
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    result = {
        "device": str(fabric.device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "particles": particles,
        "batch_size": batch_size,
        "mean_step_ms": measurement.mean * 1_000,
        "median_step_ms": measurement.median * 1_000,
        "iqr_step_ms": measurement.iqr * 1_000,
        "measurements": len(measurement.times),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
