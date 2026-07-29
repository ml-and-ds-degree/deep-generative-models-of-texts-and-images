# Compute feasibility

## What the original work reports

The IWAE paper and its official repository describe a GPU implementation but
do not identify the GPU model. Any claim about the authors' exact hardware or
wall-clock training time would therefore be speculation.

The selected one-layer network has 425,284 trainable parameters. Its memory
footprint is modest even with K=50; the paper's 3,280 passes are expensive
because they contain many optimizer steps, not because the model is large.

## M1 measurement

Measured on the target M1 Mac, macOS 15.7.7, PyTorch 2.13.0, float32. The
updated benchmark synchronizes MPS after every timed training step:

| Objective | Batch | K | Mean step | Median step | Model time/pass |
|---|---:|---:|---:|---:|---:|
| IWAE | 20 | 50 | 7.06 ms | 6.89 ms | 21.05 s |
| DReG | 20 | 50 | 7.01 ms | 6.98 ms | 20.90 s |
| DReG fast warm-up | 100 | 5 | 6.37 ms | 6.46 ms | 3.80 s |
| DReG fast final | 100 | 50 | 17.96 ms | 17.85 ms | 10.71 s |

Command:

```bash
uv run iwae benchmark --objective dreg --accelerator mps \
  --particles 50 --batch-size 20 \
  --min-run-time 1
```

There are `59,600 / 20 = 2,980` optimizer steps per pass and 9,774,400 steps in
the complete 3,280-pass schedule. Multiplying by the synchronized IWAE mean
gives about 19.2 hours of model work before validation, logging, checkpointing,
and Trainer overhead. This is an estimate, not a promised runtime; the completed
121-pass DReG run took approximately 71.5 active minutes end to end.

## Practical budgets

The schedule boundaries make principled truncated experiments possible:

| Maximum pass | Approximate M1 optimizer time | Use |
|---:|---:|---|
| 40 | 14 minutes | pipeline/debug comparison |
| 121 | 42 minutes | model-only projection; observed end-to-end was 71.5 min |
| 364 | 2.1 hours | stronger local pilot |
| 3,280 | 19.2 hours minimum | full paper schedule |

The estimate assumes the measured mean and scales linearly. Run both IWAE and
DReG to the same boundary; a shorter baseline is not a controlled comparison.

The non-reproduction `train-fast` profile projects roughly 10-11 minutes of
synchronized model work and completed end to end in 14.0 minutes for 121
passes. It is a progressive-particle result, not a paper-equivalent compute
estimate; see
[`training-acceleration.md`](training-acceleration.md) for its evidence and
quality gate.

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
