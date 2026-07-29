# Training acceleration decision

## Decision

The strongest direction for this repository is **progressive-particle DReG**:
train the inexpensive early representation with a small importance-sample
count, increase the count as optimization matures, and reserve full `K=50`
for the final refinement. The new `train-fast` profile combines this curriculum
with a larger batch and cosine learning-rate decay. The paper-faithful `train`
command is unchanged.

The seed-236 profile is now validated for the matched 121-pass experiment: it
trained in 14.0 minutes and improved full-test NLL. It remains single-seed
evidence; a population-level claim requires the replication gate below.

## Why this targets the real bottleneck

The model has only 425,284 parameters. Its long runtime comes from 2,980
optimizer updates per pass and 50 decoder evaluations per image, repeated over
the original 3,280-pass schedule. There is little value in optimizing MNIST
loading: a measured full data pass takes about 0.3 seconds, versus about 21
seconds of synchronized `K=50` model work at batch 20.

Synchronized measurements on the target M1, PyTorch 2.13.0, float32:

| Objective | Batch | K | Mean step | Estimated model time/pass |
|---|---:|---:|---:|---:|
| IWAE | 20 | 50 | 7.06 ms | 21.05 s |
| DReG | 20 | 50 | 7.01 ms | 20.90 s |
| DReG | 100 | 5 | 6.37 ms | 3.80 s |
| DReG | 100 | 50 | 17.96 ms | 10.71 s |

The benchmark synchronizes MPS/CUDA before returning each timed step. The old
benchmark did not do so, which made asynchronous accelerator timings harder to
interpret. Reproduce the table with:

```bash
uv run iwae benchmark --objective dreg --accelerator mps \
  --particles 5 --batch-size 100
uv run iwae benchmark --objective dreg --accelerator mps \
  --particles 50 --batch-size 100
```

At the default 121-pass fast schedule, K is 5 for passes 1-60, 10 for passes
61-100, and 50 for passes 101-121. The synchronized model-only projection is
roughly 10-11 minutes, about four times less than fixed `K=50`, batch-20 DReG.
End-to-end speedup can differ because logging, checkpointing, validation, and
host/device behavior are not included in that projection.

The completed end-to-end run took 838.4 seconds (14.0 minutes), selected pass
117, and produced:

| Metric | Fixed-K DReG | Progressive DReG |
|---|---:|---:|
| Best validation `-L_50` | 89.661 | **88.278** |
| Test `-L_5000` | 87.480 | **86.733** |
| Active units | 26 | **30** |
| KID (mean ± subset std.) | 206.80 ± 26.91 | **178.08 ± 21.76** |
| Training time | ~71.5 min | **14.0 min** |

This is a 5.1x end-to-end speedup and a 0.747-nat test-NLL improvement for the
shared seed and evaluation protocol.

## Pilot evidence

A matched seed-236 MPS pilot used batch 100, learning rate 0.003, DReG, and
`K=50` validation after every pass:

| Training particles | Passes | Best validation `-L_50` | Time |
|---|---:|---:|---:|
| fixed K=50 | 15 | 98.345 | 165.7 s |
| K=5 for 10, then K=50 | 15 | **96.971** | **50.4 s to best**, 94.5 s total |

The curriculum reached a substantially better validation bound in less than
one third of the fixed-K time-to-best. Later fixed-learning-rate refinement
regressed, which is why the executable profile uses cosine decay rather than
claiming that the pilot schedule itself is final.

An end-to-end 30-pass run of the implemented profile then validated the full
Lightning path:

| Run | First pass at `-L_50 <= 90.724` | Time evidence |
|---|---:|---:|
| progressive `train-fast` | 30 | 205.1 s end to end |
| existing fixed-K DReG | 60 | at least 20.9 min model-only |

The fixed-K projection is 60 multiplied by the synchronized 20.90-second model
pass, so it excludes framework overhead and is deliberately conservative. The
result establishes a greater-than-6x time-to-validation improvement at this
quality point. It does not yet establish equal final test NLL; `L_5000` remains
the acceptance metric.

Run the complete candidate with:

```bash
uv run iwae train-fast --accelerator mps --output-dir outputs
```

For a CUDA accelerator, `--precision 16-mixed` is an optional systems ablation.
It must be reported as a separate condition because the baseline uses float32.

## Quality gate

The seed-236 result passes the single-run gate. Before claiming reliable
superiority, compare it with fixed-K DReG using seeds 236, 237, and 238 and
require all of the following:

1. report end-to-end wall-clock time and time to the selected checkpoint;
2. select checkpoints only by the shared `K=50` validation metric;
3. evaluate both with the same fixed data and `L_5000` test command;
4. report mean and per-seed test NLL, active units, and KID;
5. accept the speed claim only if mean test NLL is no worse by more than a
   predeclared tolerance, suggested at 0.25 nat.

The candidate can be resumed from `outputs/fast-dreg/checkpoints/last.ckpt`.
Its checkpoint records the particle counts, boundaries, cosine horizon,
precision, and learning rates.

## Alternatives investigated

### Muon

PyTorch 2.13 includes the matrix-oriented
[Muon optimizer](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).
A direct five-pass test was rejected: it took 136.6 seconds versus 107.2 for
Adam and ended at 107.974 validation nats versus 99.606. Results reported for
large transformers do not generalize automatically to this small tanh VAE.

### Large batch alone

Batch 100 approximately halves fixed-K time per data pass, but the fixed-K
pilot learned less per pass and did not produce a drastic time-to-quality win.
It becomes useful when paired with cheap early particles, which keep each
large-batch pass inexpensive while retaining a healthy early inference signal.

### Low-rank decoder projection

The trained output matrix is not low-rank enough. Rank 50 retains 77.65% of
its squared singular-value mass but worsened a held validation-batch bound by
11.59 nats; rank 100 still worsened it by 0.99 nat and saves too little of the
whole step to justify the architectural change.

### Randomized quasi-Monte Carlo

[QMCVI](https://proceedings.mlr.press/v80/buchholz18a.html) can reduce gradient
variance for Monte Carlo variational inference. It is not a clean substitution
for the K samples *inside* IWAE's logarithm: correlating those particles changes
the joint expectation and can invalidate the standard lower-bound argument.
The QMCVI analysis also notes that minibatch sampling can dominate gradient
variance. It remains interesting for a separately derived estimator, not as a
shortcut in the controlled experiment.

### Compiler and mixed precision

[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
is worth benchmarking on the final CUDA runtime, but current MPS execution does
not offer the same Triton/CUDA graph fusion path. Mixed precision can improve
CUDA throughput, but it cannot reduce the number of required optimization
updates and must pass finite-loss and final-NLL checks. Both are secondary to
the algorithmic particle curriculum.

## Research basis

The direction is consistent with three established observations:

- tighter IWAE bounds can reduce inference-network gradient signal-to-noise;
  [Rainforth et al.](https://arxiv.org/abs/1802.04537) show that looser bounds
  can be preferable for inference learning;
- staged VAE-to-IWAE training has been used because optimizing the inexpensive
  ELBO before the IW-ELBO is cheaper than IWAE from scratch
  ([Liu et al.](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.935419/full));
- increasing the sample count over optimization is a principled continuation
  strategy for Monte Carlo objectives in
  [Buchholz et al.](https://proceedings.mlr.press/v80/buchholz18a.html).

The implementation does not claim those papers prove this exact DReG schedule.
The repository-specific pilot is the reason to pursue it; the full quality gate
is what can establish the final claim.
