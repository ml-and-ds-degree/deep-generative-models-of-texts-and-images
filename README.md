# IWAE reconstruction and DReG improvement

This project reproduces the one-stochastic-layer Importance Weighted
Autoencoder (IWAE) from Burda, Grosse, and Salakhutdinov (2015), then changes
only its encoder gradient estimator to the doubly reparameterized gradient
(DReG) from Tucker et al. (2018).

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

## Fast local verification

These commands exercise both estimators without claiming paper-level results:

```bash
uv run python -m unittest discover -s tests -v
uv run iwae train iwae --fast-dev-run --accelerator mps
uv run iwae train dreg --fast-dev-run --accelerator mps
```

Use `--accelerator cpu` if MPS is unavailable. Benchmark the actual machine
before estimating a full run. `--fast-dev-run` is Lightning's integration
check; `--limit-train-batches` and `--limit-val-batches` remain available for
deliberately truncated experiments.

```bash
uv run iwae benchmark --accelerator mps --particles 50
```

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

The complete controlled comparison and report table are in
[`docs/experiment-protocol.md`](docs/experiment-protocol.md). The measured M1
budget and Colab guidance are in
[`docs/compute-feasibility.md`](docs/compute-feasibility.md). The distinction
between faithful paper choices, engineering modernizations, evaluation
additions, and the actual DReG improvement is documented in
[`docs/paper-fidelity.md`](docs/paper-fidelity.md).

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
