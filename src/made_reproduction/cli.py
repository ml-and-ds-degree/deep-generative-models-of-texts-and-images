"""Cyclopts interface for reproducible MADE training, evaluation, and sampling."""

from __future__ import annotations

import json
import platform
from time import perf_counter
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch
from cyclopts import App
from lightning import Fabric
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from made_reproduction.config import (
    ArchitectureName,
    ATTENTION_MNIST_PRESET,
    DEEP_MNIST_HIDDEN_DIMS,
    DatasetName,
    DIRECT_MNIST_PRESET,
    RESIDUAL_MNIST_PRESET,
    paper_preset,
)
from made_reproduction.data import MADEDataModule
from made_reproduction.metrics import unbiased_rbf_mmd
from made_reproduction.module import MADELitModule

app = App(
    name="made",
    help="Reproduce the binary MADE experiments of Germain et al. (2015).",
)

Accelerator = Literal["auto", "cpu", "mps", "gpu", "tpu"]
Precision = Literal["32-true", "16-mixed", "bf16-mixed"]


def _resolved(value, default):
    return default if value is None else value


def _manifest(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **values,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "lightning": L.__version__,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_reproduction_checkpoint(
    model: MADELitModule,
    *,
    dataset: DatasetName,
    input_dim: int,
    seed: int,
) -> None:
    """Prevent a resume command from silently changing paper controls."""

    preset = paper_preset(dataset)
    expected = {
        "input_dim": input_dim,
        "hidden_dims": preset.hidden_dims,
        "activation": preset.activation.value,
        "direct_input_to_output": preset.direct_input_to_output,
        "mask_seed": seed,
        "mask_mode": preset.mask_mode.value,
        "validation_masks": preset.validation_masks,
        "test_masks": preset.test_masks,
        "optimizer": preset.optimizer.value,
        "learning_rate": preset.learning_rate,
        "decay": preset.decay,
        "epsilon": preset.epsilon,
    }
    for name, expected_value in expected.items():
        observed = model.hparams[name]
        if name == "hidden_dims":
            observed = tuple(observed)
        if observed != expected_value:
            raise ValueError(
                f"checkpoint {name}={observed!r} does not match "
                f"the {dataset.value} paper preset ({expected_value!r})"
            )


@app.command
def download(
    dataset: DatasetName = DatasetName.BINARIZED_MNIST,
    *,
    data_dir: Path = Path("data"),
) -> None:
    """Download and checksum the original prepared dataset."""

    data = MADEDataModule(dataset=dataset, data_dir=data_dir)
    data.prepare_data()
    print(data.dataset_path.resolve())


@app.command
def train(
    dataset: DatasetName = DatasetName.BINARIZED_MNIST,
    *,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs/made"),
    batch_size: int | None = None,
    num_workers: int = 0,
    seed: int | None = None,
    max_epochs: int = -1,
    patience: int | None = None,
    accelerator: Accelerator = "auto",
    precision: Precision = "32-true",
    resume_from: Path | None = None,
    fast_dev_run: bool = False,
    architecture: ArchitectureName = ArchitectureName.PAPER,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> None:
    """Train paper MADE or a binarized-MNIST architecture-improvement candidate."""

    preset = paper_preset(dataset)
    resolved_batch_size = _resolved(batch_size, preset.batch_size)
    resolved_seed = _resolved(seed, preset.seed)
    resolved_patience = _resolved(patience, preset.patience)
    if resolved_batch_size < 1 or resolved_patience < 1:
        raise ValueError("batch_size and patience must be positive")
    if max_epochs == 0 or max_epochs < -1:
        raise ValueError("max_epochs must be -1 or positive")
    if architecture is not ArchitectureName.PAPER and dataset is not DatasetName.BINARIZED_MNIST:
        raise ValueError("architecture improvements are specified only for binarized MNIST")

    L.seed_everything(resolved_seed, workers=True)
    data = MADEDataModule(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=resolved_batch_size,
        num_workers=num_workers,
        seed=resolved_seed,
    )
    data.prepare_data()
    data.setup("fit")
    resolved_optimizer = preset.optimizer
    resolved_learning_rate = preset.learning_rate
    num_heads = 4
    dropout = 0.0
    residual_blocks = 0
    hidden_to_output_skips = False
    zero_init_final_branch = False
    if architecture is ArchitectureName.PAPER:
        hidden_dims = preset.hidden_dims
        direct_input_to_output = preset.direct_input_to_output
    elif architecture is ArchitectureName.DIRECT:
        hidden_dims = (DIRECT_MNIST_PRESET.hidden_dim,)
        direct_input_to_output = DIRECT_MNIST_PRESET.direct_input_to_output
    elif architecture is ArchitectureName.DEEP:
        hidden_dims = DEEP_MNIST_HIDDEN_DIMS
        direct_input_to_output = True
        hidden_to_output_skips = True
        zero_init_final_branch = True
    elif architecture is ArchitectureName.LMCONV:
        hidden_dims = (32,)
        direct_input_to_output = True
        residual_blocks = 3
    elif architecture is ArchitectureName.PIXELCNN:
        hidden_dims = (32,)
        direct_input_to_output = True
        residual_blocks = 4
    elif architecture is ArchitectureName.ATTENTION:
        hidden_dims = (ATTENTION_MNIST_PRESET.hidden_dim,)
        direct_input_to_output = ATTENTION_MNIST_PRESET.direct_input_to_output
        residual_blocks = ATTENTION_MNIST_PRESET.residual_blocks
        num_heads = ATTENTION_MNIST_PRESET.num_heads
        dropout = ATTENTION_MNIST_PRESET.dropout
        if ATTENTION_MNIST_PRESET.optimizer is not None:
            resolved_optimizer = ATTENTION_MNIST_PRESET.optimizer
        if ATTENTION_MNIST_PRESET.learning_rate is not None:
            resolved_learning_rate = ATTENTION_MNIST_PRESET.learning_rate
    elif architecture is ArchitectureName.RESIDUAL:
        hidden_dims = (RESIDUAL_MNIST_PRESET.hidden_dim,)
        direct_input_to_output = RESIDUAL_MNIST_PRESET.direct_input_to_output
        residual_blocks = RESIDUAL_MNIST_PRESET.residual_blocks
    else:
        raise ValueError(f"unsupported architecture {architecture.value}")
    if resume_from is None:
        model = MADELitModule(
            input_dim=data.spec.input_dim,
            hidden_dims=hidden_dims,
            architecture=architecture,
            residual_blocks=residual_blocks,
            hidden_to_output_skips=hidden_to_output_skips,
            zero_init_final_branch=zero_init_final_branch,
            activation=preset.activation,
            direct_input_to_output=direct_input_to_output,
            mask_seed=resolved_seed,
            mask_mode=preset.mask_mode,
            validation_masks=preset.validation_masks,
            test_masks=preset.test_masks,
            optimizer=resolved_optimizer,
            learning_rate=resolved_learning_rate,
            decay=preset.decay,
            epsilon=preset.epsilon,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        model = MADELitModule.load_from_checkpoint(resume_from)
        if architecture is ArchitectureName.PAPER:
            _validate_reproduction_checkpoint(
                model,
                dataset=dataset,
                input_dim=data.spec.input_dim,
                seed=resolved_seed,
            )

    run_name = dataset.value if architecture is ArchitectureName.PAPER else f"{dataset.value}_{architecture.value}"
    run_dir = output_dir / run_name
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch-{epoch:04d}",
        auto_insert_metric_name=False,
        monitor="val/nll",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val/nll",
        mode="min",
        patience=resolved_patience,
        min_delta=0.0,
        check_on_train_epoch_end=False,
    )
    previous_training_time = 0.0
    existing_manifest = run_dir / "run_config.json"
    if resume_from is not None and existing_manifest.exists():
        previous_values = json.loads(existing_manifest.read_text(encoding="utf-8"))
        previous_training_time = float(previous_values.get("training_time_seconds", 0.0))
    manifest_values = {
            "dataset": dataset.value,
            "dataset_sha256": data.spec.sha256,
            "architecture": architecture.value,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "residual_blocks": residual_blocks,
            "hidden_to_output_skips": hidden_to_output_skips,
            "zero_init_final_branch": zero_init_final_branch,
            "seed": resolved_seed,
            "batch_size": resolved_batch_size,
            "max_epochs": max_epochs,
            "patience": resolved_patience,
            "hidden_dims": list(hidden_dims),
            "activation": preset.activation.value,
            "optimizer": resolved_optimizer.value,
            "learning_rate": resolved_learning_rate,
            "decay": preset.decay,
            "epsilon": preset.epsilon,
            "mask_mode": preset.mask_mode.value,
            "validation_masks": preset.validation_masks,
            "test_masks": preset.test_masks,
            "direct_input_to_output": direct_input_to_output,
            "num_heads": num_heads,
            "dropout": dropout,
            "paper_test_nll": preset.paper_test_nll,
            "paper_test_nll_ci95": preset.paper_test_nll_ci95,
    }
    _manifest(
        run_dir / "run_config.json",
        manifest_values,
    )
    trainer_arguments: dict = {
        "accelerator": accelerator,
        "devices": 1,
        "precision": precision,
        "max_epochs": max_epochs,
        "deterministic": True,
        "default_root_dir": run_dir,
        "callbacks": [checkpoint, early_stopping],
        "fast_dev_run": fast_dev_run,
    }
    if architecture is ArchitectureName.ATTENTION:
        trainer_arguments["logger"] = CSVLogger(run_dir, name="csv")
    if limit_train_batches is not None:
        trainer_arguments["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_arguments["limit_val_batches"] = limit_val_batches
    trainer = L.Trainer(**trainer_arguments)
    started = perf_counter()
    trainer.fit(model, datamodule=data, ckpt_path=resume_from)
    manifest_values["training_time_seconds"] = previous_training_time + perf_counter() - started
    manifest_values["epochs_completed"] = trainer.current_epoch
    _manifest(run_dir / "run_config.json", manifest_values)


@app.command
def evaluate(
    checkpoint: Path,
    dataset: DatasetName = DatasetName.BINARIZED_MNIST,
    *,
    data_dir: Path = Path("data"),
    batch_size: int | None = None,
    num_workers: int = 0,
    seed: int | None = None,
    masks: int | None = None,
    accelerator: Accelerator = "auto",
    limit_test_batches: int | None = None,
) -> None:
    """Evaluate exact test NLL and print the paper comparison as JSON."""

    preset = paper_preset(dataset)
    resolved_batch_size = _resolved(batch_size, preset.batch_size)
    resolved_seed = _resolved(seed, preset.seed)
    resolved_masks = _resolved(masks, preset.test_masks)
    L.seed_everything(resolved_seed, workers=True)
    data = MADEDataModule(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=resolved_batch_size,
        num_workers=num_workers,
        seed=resolved_seed,
    )
    model = MADELitModule.load_from_checkpoint(checkpoint, test_masks=resolved_masks)
    if model.hparams.input_dim != data.spec.input_dim:
        raise ValueError("checkpoint input dimension does not match the selected dataset")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        deterministic=True,
        logger=False,
        limit_test_batches=limit_test_batches,
    )
    results = trainer.test(model, datamodule=data)
    observed = results[0].get("test/nll") if results else None
    print(
        json.dumps(
            {
                "dataset": dataset.value,
                "test_nll": observed,
                "paper_test_nll": preset.paper_test_nll,
                "paper_test_nll_ci95": preset.paper_test_nll_ci95,
                "masks": resolved_masks,
            },
            indent=2,
        )
    )


@app.command
def sample(
    checkpoint: Path,
    *,
    output: Path = Path("samples.npz"),
    count: int = 1_000,
    seed: int = 1234,
    accelerator: Accelerator = "auto",
) -> None:
    """Save reproducible ancestral binary samples from a checkpoint."""

    if count < 1:
        raise ValueError("count must be positive")
    L.seed_everything(seed, workers=True)
    fabric = Fabric(accelerator=accelerator, devices=1, precision="32-true")
    fabric.launch()
    model = MADELitModule.load_from_checkpoint(checkpoint).eval()
    model = fabric.setup_module(model)
    with torch.inference_mode():
        # ``seed_everything`` seeds the active backend. MPS does not expose a
        # user-constructible device Generator, so use its seeded global stream.
        samples = model.model.sample(count).cpu().numpy().astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, samples=samples, seed=np.asarray(seed, dtype=np.int64))
    print(output.resolve())


@app.command
def mmd(
    checkpoint: Path,
    dataset: DatasetName = DatasetName.BINARIZED_MNIST,
    *,
    data_dir: Path = Path("data"),
    count: int = 2_000,
    bandwidth_squared: float = 196.0,
    seed: int = 12_345,
    accelerator: Accelerator = "auto",
) -> None:
    """Estimate unbiased RBF-MMD against a fixed held-out binary-MNIST subset."""

    if count < 2:
        raise ValueError("count must be at least two")
    L.seed_everything(seed, workers=True)
    data = MADEDataModule(dataset=dataset, data_dir=data_dir, seed=seed)
    data.prepare_data()
    data.setup("test")
    if data.test_data is None or count > len(data.test_data):
        raise ValueError("count exceeds the available held-out split")
    fabric = Fabric(accelerator=accelerator, devices=1, precision="32-true")
    fabric.launch()
    model = MADELitModule.load_from_checkpoint(checkpoint).eval()
    model = fabric.setup_module(model)
    with torch.inference_mode():
        generated = model.model.sample(count).cpu()
    reference = data.test_data.values[:count]
    value = unbiased_rbf_mmd(
        generated, reference, bandwidth_squared=bandwidth_squared
    ).item()
    print(
        json.dumps(
            {
                "dataset": dataset.value,
                "architecture": model.hparams.architecture,
                "metric": "unbiased_rbf_mmd_squared",
                "mmd_squared": value,
                "sample_count_per_set": count,
                "bandwidth_squared": bandwidth_squared,
                "seed": seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
