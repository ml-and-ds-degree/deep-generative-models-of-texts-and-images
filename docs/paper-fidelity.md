# Paper fidelity and modern implementation notes

This project contains three different kinds of changes. They must not be
conflated in the report:

1. **Faithful reconstruction choices** reproduce Burda et al. (2015).
2. **Engineering modernizations** use current PyTorch and Lightning APIs but do
   not change the model, objective, or intended optimization trajectory.
3. **Experimental additions or improvements** either add a course metric,
   strengthen reproducibility, or constitute the Stage-2 scientific change.

Only DReG is the claimed Stage-2 scientific improvement.

## Classification of implementation choices

| Implementation choice | Classification | Effect on the experiment |
|---|---|---|
| One-layer 784-200-200-50 encoder/decoder | Faithful reconstruction | Matches the selected paper architecture |
| Tanh hidden activations and Bernoulli decoder | Faithful reconstruction | Matches the paper likelihood model |
| Xavier initialization and empirical pixel-logit bias | Faithful reconstruction | Matches the reported initialization procedure |
| Adam β=(0.9, 0.999), ε=1e-4, batch 20 | Faithful reconstruction | Matches the paper optimizer settings |
| K=50 IWAE objective | Faithful reconstruction | Matches the selected paper condition |
| `MultiStepLR` at passes 1, 4, 13, 40, 121, 364, 1093 | Engineering modernization | Expresses the paper schedule exactly through PyTorch |
| Lightning `ModelCheckpoint` and resume state | Engineering modernization | Adds reliable interruption recovery; does not change the objective |
| Lightning manual optimization | Engineering necessity | Required to apply different encoder and decoder gradient estimators |
| Two Adam optimizer objects | Engineering necessity | Separates encoder and decoder gradients for DReG; Adam state is parameter-local and both are stepped once per batch |
| TorchMetrics accumulation | Engineering modernization | Adds tested device placement and distributed reduction |
| Lightning Fabric for sampling/KID | Engineering modernization | Replaces handwritten CPU/CUDA/MPS selection and extends portability |
| `torch.utils.benchmark` | Engineering modernization | Improves timing methodology only |
| Chunked L_5000 | Exact computational reformulation | Produces the same log-sum-exp estimator while bounding memory |
| Fixed validation/test Bernoulli draws | Reproducibility addition | Enables paired comparisons, but differs from repeatedly resampling dynamically binarized evaluation images |
| MNIST-classifier-feature KID | Course metric addition | Not reported by the IWAE paper and not comparable to ImageNet-Inception KID |
| DReG encoder estimator | **Stage-2 scientific improvement** | Changes the encoder gradient estimator while holding the generative objective and architecture fixed |

## Important evaluation deviation

The original implementation dynamically binarizes MNIST when examples are
retrieved. This project keeps training binarization dynamic but deterministically
fixes one Bernoulli draw per validation/test example. That design reduces
between-run evaluation noise and ensures IWAE and DReG see exactly the same
observations.

This is a reproducibility improvement, not a claim that the paper used fixed
evaluation images. The report should state it explicitly. If strict replication
of the original evaluation sampling is required, add a dynamic-evaluation mode
and report both protocols rather than silently changing the existing comparison.

## KID interpretation

The KID command uses TorchMetrics' unbiased polynomial-kernel MMD implementation
with the project's fixed MNIST classifier as the feature extractor. It reports:

- `kid_mean`: mean KID across random subsets;
- `kid_std`: standard deviation across subset estimates.

The original IWAE paper did not report KID. The value is only comparable between
runs that share the classifier checkpoint, real test examples, sample count,
subset size, number of subsets, and random seed.

## What remains deliberately handwritten

PyTorch does not provide an IWAE or DReG objective, the paper's active-unit
definition, or a chunked L_5000 evaluator. These remain explicit research code
so their equations can be inspected against the papers. The diagonal Gaussian
and Bernoulli terms also remain vectorized explicitly because DReG requires a
separate path where the posterior parameters are detached while the latent
sample remains differentiable.

## Suggested report wording

> We reconstructed the one-layer K=50 IWAE architecture and optimization
> protocol of Burda et al. using PyTorch 2.13 and PyTorch Lightning 2.6.
> Framework-native schedulers, checkpointing, distributed metric accumulation,
> and accelerator handling are engineering modernizations and are not treated
> as model improvements. Our Stage-2 change is restricted to the DReG encoder
> gradient estimator. For reproducible paired evaluation, validation and test
> Bernoulli draws are fixed per example; this deviation from dynamic evaluation
> is reported separately.
