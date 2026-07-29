# IWAE reconstruction and DReG improvement

This project reproduces the one-stochastic-layer Importance Weighted
Autoencoder (IWAE) from Burda, Grosse, and Salakhutdinov (2015), then changes
only its encoder gradient estimator to the doubly reparameterized gradient
(DReG) from Tucker et al. (2018).

It also includes an explicitly non-reproduction `train-fast` profile that
uses progressive particle counts. On the completed seed-236 run it reduced
training from approximately 71.5 to 14.0 minutes and improved test
`-L_5000` from 87.480 to 86.733. Under the shared 10,000-sample KID protocol,
it also improved KID from `206.80 ± 26.91` to `178.08 ± 21.76` (lower is
better). Its rationale, rejected alternatives, and multi-seed quality gate are in
[`docs/training-acceleration.md`](docs/training-acceleration.md).

The scope is sized for development on an M1 Mac with 16 GB unified memory and
final training on a Colab GPU. The paper's complete 3,280-pass schedule is
computationally long but memory-safe; short runs are available for smoke tests.

## What is reproduced

- dynamically binarized MNIST with the original 59,600/400/10,000 split;
- encoder `784 → 200 → 200 → (μ₅₀, log σ₅₀)` with tanh activations;
- decoder `50 → 200 → 200 → 784` with tanh hidden activations and Bernoulli
  output;
- Xavier initialization and empirical pixel-logit decoder bias;
- Adam with β₁=0.9, β₂=0.999, ε=1e-4 and batch size 20;
- K=50 training objective and the eight-stage 3,280-pass learning-rate schedule;
- memory-bounded L_5000 test NLL and active-latent-unit evaluation.

The improvement keeps all of those controls fixed and replaces the original
IWAE inference-network estimator with DReG.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync
uv run iwae --help
```

MNIST is downloaded automatically into `data/` on first use.

## Dev Container and MPS workflow

> [!WARNING]
> A Dev Container runs Linux, so it cannot access macOS Metal Performance
> Shaders (MPS). Use `--accelerator cpu` inside the container, or run training
> natively on macOS when MPS acceleration is required.

The Dev Container provides the locked Python environment plus `latexmk` and
`pdftoppm` for reproducible report work. Its virtual environment is separate
from the local `.venv`, but the repository itself is shared, so checkpoints,
metrics, figures, and report sources remain available in both environments.

For the usual hybrid workflow:

1. On the macOS host, train with MPS and write artifacts into the shared
   repository:

   ```bash
   uv sync
   uv run iwae train iwae --accelerator mps --output-dir outputs/mps
   ```

2. Reopen the repository with **Dev Containers: Reopen in Container**. Build
   the report with LaTeX Workshop, or run:

   ```bash
   cd report
   latexmk -pdf -outdir=build main.tex
   ```

The container is also suitable for CPU-only tests, for example
`uv run python -m unittest discover -s tests -v`.

## Fast local verification

These commands exercise both estimators without claiming paper-level results:

```bash
uv run python -m unittest discover -s tests -v
uv run iwae train iwae --fast-dev-run --accelerator mps
uv run iwae train dreg --fast-dev-run --accelerator mps
uv run iwae train-fast --fast-dev-run --accelerator mps
```

Use `--accelerator cpu` if MPS is unavailable. Benchmark the actual machine
before estimating a full run. `--fast-dev-run` is Lightning's integration
check; `--limit-train-batches` and `--limit-val-batches` remain available for
deliberately truncated experiments.

```bash
uv run iwae benchmark --accelerator mps --particles 50
```

Run the accelerated candidate separately from the controlled reproduction:

```bash
uv run iwae train-fast --accelerator mps --output-dir outputs
```

The default candidate uses DReG with K=5, then K=10, then K=50; cosine learning
rate decay; batch size 100; and K=50 validation throughout. Artifacts are kept
under `outputs/fast-dreg/` so they cannot overwrite the fixed-K DReG run.

## Full controlled runs

Run the reconstruction and improvement with the same seed and budget:

```bash
uv run iwae train iwae --accelerator gpu --seed 236 --output-dir outputs/full
uv run iwae train dreg --accelerator gpu --seed 236 --output-dir outputs/full
```

Resume an interrupted run using its `last.ckpt`:

```bash
uv run iwae train iwae --accelerator gpu \
  --resume-from outputs/full/iwae/checkpoints/last.ckpt \
  --output-dir outputs/full
```

Evaluate the best checkpoint with the paper's estimator:

```bash
uv run iwae evaluate CHECKPOINT.ckpt --accelerator gpu \
  --particles 5000 --chunk-size 100
uv run iwae sample CHECKPOINT.ckpt --accelerator gpu --output samples.png
```

For the course-aligned KID metric, first train the fixed MNIST feature
classifier once, then reuse its best checkpoint for both generative models.
The command uses TorchMetrics and reports subset mean and standard deviation:

```bash
uv run iwae train-classifier --accelerator gpu
uv run iwae kid MODEL.ckpt CLASSIFIER.ckpt --accelerator gpu
```

The measured M1 budget and Colab guidance are in
[`docs/compute-feasibility.md`](docs/compute-feasibility.md). Experimental
invariants and paper-fidelity rationale are kept as comments beside the
relevant data, model, objective, metric, and CLI implementation.

## Reproducibility

Every command seeds Python/NumPy/PyTorch through Lightning, seeds workers, and
requests deterministic trainer behavior. Training MNIST is resampled
dynamically as required by the paper. Validation and test Bernoulli draws are
fixed per example so model comparisons use identical observations. Exact
bitwise agreement is not guaranteed across PyTorch releases or CPU, MPS, CUDA,
and TPU backends.

The main result should record the checkpoint, seed, accelerator, package lock,
training time, test NLL, active units, and KID. The paper target for the selected
one-layer K=50 model is **84.78 nats** and **25 active units**.

## Sources

- [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)
- [Official IWAE implementation](https://github.com/yburda/iwae)
- [Doubly Reparameterized Gradient Estimators](https://arxiv.org/abs/1810.04152)
- [Stanford CS236 syllabus](https://deepgenerativemodels.github.io/syllabus.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Lightning reproducibility API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility)
- [Cyclopts documentation](https://cyclopts.readthedocs.io/en/stable/)
