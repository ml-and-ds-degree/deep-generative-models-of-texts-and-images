"""Build the static figures used by the MADE reproduction report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

INK = "#20252b"
BLUE = "#3568a8"
ORANGE = "#c77a27"
GRID = "#d9dde2"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
        }
    )


def _read_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    epochs: dict[int, dict[str, float]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                epoch = int(row["epoch"])
                values = epochs.setdefault(epoch, {})
                if row["train/nll"]:
                    values["train"] = float(row["train/nll"])
                if row["val/nll"]:
                    values["valid"] = float(row["val/nll"])
    ordered = sorted(epoch for epoch, values in epochs.items() if len(values) == 2)
    return {
        "epoch": np.asarray(ordered),
        "train": np.asarray([epochs[epoch]["train"] for epoch in ordered]),
        "valid": np.asarray([epochs[epoch]["valid"] for epoch in ordered]),
    }


def training_figure(metrics: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 2.35), constrained_layout=True)
    axis.plot(
        metrics["epoch"],
        metrics["train"],
        color=BLUE,
        linewidth=1.6,
        label="Train",
    )
    axis.plot(
        metrics["epoch"],
        metrics["valid"],
        color=ORANGE,
        linewidth=1.6,
        linestyle="--",
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("NLL (nats per example; lower is better)")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right", ncols=2)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def samples_figure(samples_path: Path, output: Path) -> None:
    archive = np.load(samples_path)
    samples = archive["samples"][:100].reshape(10, 10, 28, 28)
    canvas = np.ones((10 * 28 + 9, 10 * 28 + 9), dtype=np.uint8)
    for row in range(10):
        for column in range(10):
            y = row * 29
            x = column * 29
            canvas[y : y + 28, x : x + 28] = samples[row, column]
    fig, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    axis.imshow(
        canvas,
        interpolation="nearest",
        cmap=ListedColormap(["#171a1f", "#f4f6f8"]),
        vmin=0,
        vmax=1,
    )
    axis.set_axis_off()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--improved-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--improved-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    training_figure(
        _read_metrics(arguments.baseline_metrics),
        arguments.output_dir / "training_validation.pdf",
    )
    samples_figure(
        arguments.baseline_samples,
        arguments.output_dir / "mnist_samples.pdf",
    )
    training_figure(
        _read_metrics(arguments.improved_metrics),
        arguments.output_dir / "pixelcnn_training_validation.pdf",
    )
    samples_figure(
        arguments.improved_samples,
        arguments.output_dir / "pixelcnn_samples.pdf",
    )


if __name__ == "__main__":
    main()
