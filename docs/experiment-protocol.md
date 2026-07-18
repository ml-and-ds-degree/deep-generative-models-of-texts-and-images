# Experiment protocol

## Research question

Can the doubly reparameterized gradient estimator (DReG) improve a faithful
one-stochastic-layer IWAE reconstruction when architecture, data, optimizer,
particle count, initialization, and training budget are held fixed?

## Controlled comparison

| Component | Reconstruction | Improvement |
|---|---|---|
| Encoder/decoder | 784-200-200-50 / 50-200-200-784 | identical |
| Training data | dynamically binarized MNIST, 59,600 examples | identical |
| Validation data | final 400 MNIST training examples, fixed Bernoulli draw | identical |
| Optimizer | Adam, β=(0.9, 0.999), ε=1e-4 | identical |
| Batch size | 20 | identical |
| Importance samples | K=50 | identical |
| Encoder gradient | original IWAE estimator | DReG |
| Primary metric | test NLL using L_5000 (nats; lower is better) | same |
| Representation metric | active units, Var(E[z|x]) > 0.01 (higher is descriptive) | same |
| Course sample metric | MNIST-classifier-feature KID mean ± subset SD (lower is better) | same |

The classifier checkpoint, deterministic test binarization seed, KID subset
indices, and evaluation seed must be shared by both runs.

## Paper reference values

For dynamically binarized MNIST, the one-layer IWAE with K=50 reports:

- test NLL: **84.78 nats**, estimated with L_5000;
- active latent dimensions: **25 of 50**.

These are comparison targets, not pass/fail thresholds. Hardware, current
PyTorch kernels, fixed evaluation binarization, and stochastic optimization can
produce differences even with a faithful reconstruction.

## Run matrix

Use seed 236 for the main comparison. If compute permits, repeat both conditions
with seeds 237 and 238 and report mean ± standard deviation. Never compare a
long DReG run with a shorter IWAE run.

1. Run `iwae train iwae ...` and `iwae train dreg ...` with identical options.
2. Evaluate each best checkpoint with exactly 5,000 particles.
3. Generate an equal-sized sample set from each checkpoint.
4. Compute KID with the same classifier checkpoint and subset seed; record the
   TorchMetrics subset mean and standard deviation.
5. Record wall-clock time, accelerator, PyTorch version, and final epoch.

## Results table template

| Model | Test NLL ↓ | Active units | KID ↓ | Training time |
|---|---:|---:|---:|---:|
| IWAE paper (K=50) | 84.78 | 25 | not reported | not reported |
| Reconstruction | TBD | TBD | TBD | TBD |
| IWAE-DReG | TBD | TBD | TBD | TBD |

## Interpretation guardrails

- NLL is the primary paper comparison; KID measures a different property.
- KID is only comparable between runs using the same learned feature extractor.
- Fixed validation/test Bernoulli draws are a paired-evaluation reproducibility
  addition, not an undocumented claim about the paper's evaluation protocol.
- A tighter training bound does not by itself establish better samples.
- The selected one-layer model is deliberate: the original DReG estimator
  applies directly, whereas hierarchical latent models require generalized
  estimators.
