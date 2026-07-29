"""Cyclopts command-line interface for every reproducible experiment.

For a controlled IWAE/DReG comparison, invoke ``train`` with identical options
and seeds, evaluate checkpoints with the same 5,000-particle configuration,
and reuse one classifier checkpoint and KID subset seed. Never compare models
trained for different budgets. The primary paper metric is test NLL (lower is
better); active units are descriptive and classifier-feature KID measures a
different property. The selected paper references are 84.78 nats and 25 active
units, not pass/fail thresholds.
"""

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
from iwae_reproduction.config import PAPER_EPOCH_BOUNDARIES, LearningRateSchedule, Objective
from iwae_reproduction.data import MNISTDataModule
from iwae_reproduction.module import IWAELitModule
from iwae_reproduction.objectives import dreg_encoder_loss, iwae_loss

app = App(
    name="iwae",
    help="Reproduce one-layer IWAE and compare the DReG encoder estimator.",
)

Accelerator = Literal["auto", "cpu", "mps", "gpu", "tpu"]
Precision = Literal["32-true", "16-mixed", "bf16-mixed"]


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
    precision: Precision = "32-true",
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
) -> L.Trainer:
    arguments: dict = {
        "accelerator": accelerator,
        "devices": 1,
        "max_epochs": max_epochs,
        "precision": precision,
        "deterministic": deterministic,
        "default_root_dir": output_dir,
        "callbacks": callbacks,
        "fast_dev_run": fast_dev_run,
        "enable_progress_bar": enable_progress_bar,
        "enable_model_summary": enable_model_summary,
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
    _run_training(
        objective=objective,
        run_name=objective.value,
        train_particles=train_particles,
        training_particle_counts=None,
        training_particle_boundaries=(),
        learning_rate=1e-3,
        learning_rate_schedule=LearningRateSchedule.PAPER,
        minimum_learning_rate=1e-4,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision="32-true",
        resume_from=resume_from,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
        enable_model_summary=True,
    )


@app.command(name="train-fast")
def train_fast(
    *,
    train_particles: int = 50,
    warmup_particles: int = 5,
    middle_particles: int = 10,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs"),
    batch_size: int = 100,
    num_workers: int = 0,
    seed: int = 236,
    max_epochs: int = 121,
    learning_rate: float = 3e-3,
    minimum_learning_rate: float = 1e-4,
    accelerator: Accelerator = "auto",
    precision: Precision = "32-true",
    resume_from: Path | None = None,
    fast_dev_run: bool = False,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> None:
    """Train DReG with progressive particles and a cosine learning rate.

    This is an explicitly non-reproduction profile. It spends the first half
    of the budget at a cheap K, the next third at an intermediate K, and the
    final sixth at the requested K. Validation always uses ``train_particles``
    so checkpoint selection remains comparable to fixed-K DReG.
    """
    counts, boundaries = _progressive_particle_schedule(
        max_epochs,
        warmup_particles,
        middle_particles,
        train_particles,
    )
    _run_training(
        objective=Objective.DREG,
        run_name="fast-dreg",
        train_particles=train_particles,
        training_particle_counts=counts,
        training_particle_boundaries=boundaries,
        learning_rate=learning_rate,
        learning_rate_schedule=LearningRateSchedule.COSINE,
        minimum_learning_rate=minimum_learning_rate,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        resume_from=resume_from,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _progressive_particle_schedule(
    max_epochs: int,
    warmup_particles: int,
    middle_particles: int,
    final_particles: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if min(warmup_particles, middle_particles, final_particles) < 1:
        raise ValueError("particle counts must be positive")
    if not warmup_particles <= middle_particles <= final_particles:
        raise ValueError("particle counts must be non-decreasing")
    if max_epochs < 6 or warmup_particles == final_particles:
        return (final_particles,), ()

    first_boundary = max_epochs // 2
    second_boundary = 5 * max_epochs // 6
    counts = (warmup_particles, middle_particles, final_particles)
    if warmup_particles == middle_particles:
        return (warmup_particles, final_particles), (second_boundary,)
    if middle_particles == final_particles:
        return (warmup_particles, final_particles), (first_boundary,)
    return counts, (first_boundary, second_boundary)


def _run_training(
    *,
    objective: Objective,
    run_name: str,
    train_particles: int,
    training_particle_counts: tuple[int, ...] | None,
    training_particle_boundaries: tuple[int, ...],
    learning_rate: float,
    learning_rate_schedule: LearningRateSchedule,
    minimum_learning_rate: float,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_epochs: int,
    accelerator: Accelerator,
    precision: Precision,
    resume_from: Path | None,
    fast_dev_run: bool,
    limit_train_batches: int | None,
    limit_val_batches: int | None,
    enable_progress_bar: bool,
    enable_model_summary: bool,
) -> None:
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
            training_particle_counts=training_particle_counts,
            training_particle_boundaries=training_particle_boundaries,
            learning_rate=learning_rate,
            learning_rate_schedule=learning_rate_schedule.value,
            schedule_epochs=max_epochs,
            minimum_learning_rate=minimum_learning_rate,
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
        expected_counts = (
            (train_particles,)
            if training_particle_counts is None
            else tuple(training_particle_counts)
        )
        if model.training_particle_counts != expected_counts:
            raise ValueError("checkpoint training-particle schedule does not match this command")
        if model.training_particle_boundaries != training_particle_boundaries:
            raise ValueError("checkpoint particle boundaries do not match this command")
        if model.learning_rate_schedule is not learning_rate_schedule:
            raise ValueError("checkpoint learning-rate schedule does not match this command")
        if model.hparams.learning_rate != learning_rate:
            raise ValueError("checkpoint learning rate does not match this command")
        if learning_rate_schedule is LearningRateSchedule.COSINE:
            if model.hparams.schedule_epochs != max_epochs:
                raise ValueError("checkpoint cosine horizon does not match --max-epochs")
            if model.hparams.minimum_learning_rate != minimum_learning_rate:
                raise ValueError("checkpoint minimum learning rate does not match this command")

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / run_name / "checkpoints",
        filename="epoch-{epoch:04d}",
        auto_insert_metric_name=False,
        monitor="val/nll_bound",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    trainer = _trainer(
        output_dir=output_dir / run_name,
        accelerator=accelerator,
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=[checkpoint],
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
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
    """Report primary test NLL (L_K) and descriptive active-unit count."""
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
    """Compute course-specific MNIST-feature KID; lower is better.

    Compare values only when both models share the classifier checkpoint,
    deterministic test data, count, subset size/count, and seed. TorchMetrics
    returns the subset mean and subset standard deviation.
    """
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
    objective: Objective = Objective.IWAE,
    particles: int = 50,
    batch_size: int = 20,
    min_run_time: float = 1.0,
    seed: int = 236,
    accelerator: Accelerator = "auto",
) -> None:
    """Measure timing only; this does not modify the scientific experiment."""
    if particles < 1 or batch_size < 1 or min_run_time <= 0:
        raise ValueError("particles, batch_size, and min_run_time must be positive")
    L.seed_everything(seed, workers=True)
    fabric = _fabric(accelerator)
    model = fabric.setup_module(IWAELitModule(train_particles=particles, objective=objective.value))
    encoder_optimizer, decoder_optimizer = fabric.setup_optimizers(
        torch.optim.Adam(model.encoder.parameters(), lr=1e-3, eps=1e-4),
        torch.optim.Adam(model.decoder.parameters(), lr=1e-3, eps=1e-4),
    )
    images = torch.bernoulli(torch.full((batch_size, 784), 0.15, device=fabric.device))
    encoder_parameters = tuple(model.encoder.parameters())
    decoder_parameters = tuple(model.decoder.parameters())

    def benchmark_step() -> None:
        encoder_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        log_weights, pathwise_log_weights = model._draw_terms(images, particles)
        bound_loss = iwae_loss(log_weights)
        if objective is Objective.IWAE:
            fabric.backward(bound_loss)
        else:
            fabric.backward(bound_loss, inputs=decoder_parameters, retain_graph=True)
            encoder_loss = dreg_encoder_loss(log_weights, pathwise_log_weights)
            fabric.backward(encoder_loss, inputs=encoder_parameters)
        encoder_optimizer.step()
        decoder_optimizer.step()
        _synchronize(fabric.device)

    timer = benchmark_utils.Timer(
        stmt="benchmark_step()",
        globals={"benchmark_step": benchmark_step},
        label=f"{objective.value.upper()} training step",
        sub_label=f"K={particles}, batch={batch_size}",
        description=str(fabric.device),
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    result = {
        "device": str(fabric.device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "objective": objective.value,
        "particles": particles,
        "batch_size": batch_size,
        "mean_step_ms": measurement.mean * 1_000,
        "median_step_ms": measurement.median * 1_000,
        "iqr_step_ms": measurement.iqr * 1_000,
        "measurements": len(measurement.times),
        "estimated_pass_seconds": measurement.mean * ((59_600 + batch_size - 1) // batch_size),
    }
    print(json.dumps(result, indent=2))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    app()
