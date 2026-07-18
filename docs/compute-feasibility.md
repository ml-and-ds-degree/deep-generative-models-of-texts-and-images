# Compute feasibility

## What the original work reports

The IWAE paper and its official repository describe a GPU implementation but
do not identify the GPU model. Any claim about the authors' exact hardware or
wall-clock training time would therefore be speculation.

The selected one-layer network has 425,284 trainable parameters. Its memory
footprint is modest even with K=50; the paper's 3,280 passes are expensive
because they contain many optimizer steps, not because the model is large.

## M1 measurement

Measured on the target M1 Mac, macOS 15.7.7, PyTorch 2.13.0, float32:

| Batch | K | Mean step | Median step |
|---:|---:|---:|---:|
| 20 | 50 | 9.47 ms | 10.00 ms |

Command:

```bash
uv run iwae benchmark --accelerator mps --particles 50 --batch-size 20 \
  --min-run-time 1
```

There are `59,600 / 20 = 2,980` optimizer steps per pass and 9,774,400 steps in
the complete 3,280-pass schedule. Multiplying by the measured mean gives about
25.7 hours before dataloading, validation, logging, and checkpoint overhead.
DReG performs two restricted backward traversals and can be somewhat slower.
This is an estimate, not a promised runtime.

## Practical budgets

The schedule boundaries make principled truncated experiments possible:

| Maximum pass | Approximate M1 optimizer time | Use |
|---:|---:|---|
| 40 | 19 minutes | pipeline/debug comparison |
| 121 | 57 minutes | one-hour local comparison |
| 364 | 2.9 hours | stronger local pilot |
| 3,280 | 25.7 hours minimum | full paper schedule |

The estimate assumes the measured mean and scales linearly. Run both IWAE and
DReG to the same boundary; a shorter baseline is not a controlled comparison.

For the final 3,280-pass result, a Colab GPU is the recommended target. Run the
benchmark command first in the allocated runtime and compute:

```text
estimated_hours = mean_step_ms × 9,774,400 / 3,600,000
```

Colab hardware allocation is variable, so record `torch.cuda.get_device_name(0)`
and the benchmark JSON with the result. Checkpointing to persistent storage is
important because a complete run can outlast a single session.

Lightning exposes TPU support, and the training command accepts
`--accelerator tpu`, but this project has only been validated on CPU and MPS.
Use a Colab GPU for the graded experiment unless the exact PyTorch/XLA
environment and manual-optimization path are smoke-tested first.
