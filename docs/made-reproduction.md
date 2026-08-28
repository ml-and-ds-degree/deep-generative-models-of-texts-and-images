# MADE reproduction protocol

This package reconstructs selected binary density-estimation experiments from
Germain et al., *MADE: Masked Autoencoder for Distribution Estimation* (ICML
2015). The default target is the one-hidden-layer, one-mask binarized-MNIST
result: it has image geometry, a precise main-table target, and fits Apple MPS.

## Controlled binarized-MNIST experiment

| Item | Paper / author code | `made train binarized-mnist` |
|---|---|---|
| Prepared split | 50,000 / 10,000 / 10,000 | matched, checksum-pinned |
| Input dimension | 784 static binary pixels | matched |
| Hidden layers | one layer of 8,000 units | matched |
| Activation | ReLU | matched |
| Autoregressive graph | one shuffled fixed mask | matched |
| Direct connection | disabled | matched |
| Initialization | orthogonal weights, zero biases | matched with PyTorch equivalent |
| Optimizer | Adagrad at 0.01, epsilon 1e-6 | matched |
| Mini-batch size | 100 | matched |
| Training order | fixed prepared-array order | matched |
| Selection | validation NLL, patience 30 | matched |
| Test metric | exact mean NLL in nats | matched |
| Paper result | 88.40 ± 0.45 | comparison target |

PyTorch, Lightning, Cyclopts, atomic dataset download, SHA-256 verification,
checkpointing, CSV logging, and the JSON run manifest are engineering
modernizations. They do not alter the likelihood, model graph, data split, or
optimizer update.

## Reproducibility controls

- The source dataset URL is pinned to the authors' ICML 2015 release and its bytes are
  checked before use.
- Python, NumPy, and PyTorch are seeded through Lightning.
- Every mask is a deterministic function of `mask_seed + mask_index`; both the
  active masks and index are checkpoint buffers.
- Fixed-order mini-batches match the author implementation and make resumed
  data traversal independent of sampler state.
- Resume commands reject checkpoints whose architecture, optimizer, seed, or
  mask controls differ from the selected paper preset.
- `run_config.json` records the resolved scientific configuration and runtime
  versions.

Reproducibility means a stable protocol, not guaranteed bitwise equality
between CPU, MPS, and CUDA linear-algebra kernels. Final comparisons should use
the same accelerator and locked environment for all models.

## Commands

Run a one-batch integration check:

```bash
uv run made train binarized-mnist --accelerator mps --fast-dev-run
```

Run the reconstruction until validation early stopping, as in the authors'
released command:

```bash
uv run made train binarized-mnist --accelerator mps
```

For a bounded pilot, set a positive epoch cap:

```bash
uv run made train binarized-mnist --accelerator mps --max-epochs 1000
```

Evaluate the best checkpoint on the complete test split:

```bash
uv run made evaluate outputs/made/binarized_mnist/checkpoints/epoch-NNNN.ckpt \
  binarized-mnist \
  --accelerator mps
```

Resume the last checkpoint without changing scientific controls:

```bash
uv run made train binarized-mnist --accelerator mps --max-epochs 2000 \
  --resume-from outputs/made/binarized_mnist/checkpoints/last.ckpt
```

## DNA preset

`made train dna` implements the author-provided mask-sampling experiment:
one 500-unit ReLU layer, an input-output skip, AdaDelta with rho 0.95 and
epsilon 1e-5, a new full mask after every update, 300-mask validation, and
1,000-mask test evaluation. Its paper comparison target is 79.66 ± 0.63 nats.
It is available for a second reconstruction condition but is not the default
because mask-ensemble validation is substantially more expensive.

## Sources

- [MADE paper](https://proceedings.mlr.press/v37/germain15.html)
- [Author implementation](https://github.com/mgermain/MADE)
- [Paper supplementary confidence intervals](https://homepages.inf.ed.ac.uk/imurray2/pub/15made/germain15-supp.pdf)
