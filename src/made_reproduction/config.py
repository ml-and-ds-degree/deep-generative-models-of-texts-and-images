"""Paper-backed configurations for the MADE reproduction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DatasetName(StrEnum):
    """Datasets with complete commands in the authors' released repository."""

    ADULT = "adult"
    BINARIZED_MNIST = "binarized_mnist"
    DNA = "dna"


class ActivationName(StrEnum):
    """Hidden activations used by the selected paper experiments."""

    TANH = "tanh"
    RELU = "relu"


class OptimizerName(StrEnum):
    """Optimizers used by the selected paper experiments and architecture studies."""

    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMW = "adamw"


class MaskMode(StrEnum):
    """Whether one fixed MADE graph or sampled graphs are trained."""

    FIXED = "fixed"
    EVERY_BATCH = "every-batch"


class ArchitectureName(StrEnum):
    """Architectures exposed by the reproducible experiment interface."""

    PAPER = "paper"
    DIRECT = "direct"
    DEEP = "deep"
    LMCONV = "lmconv"
    PIXELCNN = "pixelcnn"
    GATED_PIXELCNN = "gated-pixelcnn"
    LMCONV_ENSEMBLE = "lmconv-ensemble"
    RESIDUAL = "residual"
    ATTENTION = "attention"


@dataclass(frozen=True)
class ImprovementPreset:
    """Fixed, compute-aware architecture-improvement configuration."""

    architecture: ArchitectureName
    hidden_dim: int
    residual_blocks: int
    direct_input_to_output: bool
    num_heads: int = 4
    dropout: float = 0.0
    optimizer: OptimizerName | None = None
    learning_rate: float | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of an original, checksum-pinned dataset file."""

    name: DatasetName
    input_dim: int
    train_size: int
    validation_size: int
    test_size: int
    url: str
    sha256: str


@dataclass(frozen=True)
class PaperPreset:
    """Resolved hyperparameters from an author-provided reproduction command."""

    dataset: DatasetName
    hidden_dims: tuple[int, ...]
    activation: ActivationName
    optimizer: OptimizerName
    learning_rate: float
    decay: float
    epsilon: float
    mask_mode: MaskMode
    validation_masks: int
    test_masks: int
    batch_size: int
    patience: int
    seed: int
    direct_input_to_output: bool
    paper_test_nll: float
    paper_test_nll_ci95: float


_COMMIT = "514849eb9451311c02049f9e2d9a17fac589996a"
_RAW_DATA = f"https://raw.githubusercontent.com/mgermain/MADE/{_COMMIT}/datasets"
_RELEASE_DATA = "https://github.com/mgermain/MADE/releases/download/ICML2015"

DATASETS: dict[DatasetName, DatasetSpec] = {
    DatasetName.BINARIZED_MNIST: DatasetSpec(
        name=DatasetName.BINARIZED_MNIST,
        input_dim=784,
        train_size=50_000,
        validation_size=10_000,
        test_size=10_000,
        url=f"{_RELEASE_DATA}/binarized_mnist.npz",
        sha256="af10774d1da7e037c2f2d0388e6fd51e99f03d8878b72c15c86e0ca91c1424b2",
    ),
    DatasetName.ADULT: DatasetSpec(
        name=DatasetName.ADULT,
        input_dim=123,
        train_size=5_000,
        validation_size=1_414,
        test_size=26_147,
        url=f"{_RAW_DATA}/adult.npz",
        sha256="7ad832613f0972219961ce0060540d7c3ff324e0ec69b5ce2ad0babda9eddd10",
    ),
    DatasetName.DNA: DatasetSpec(
        name=DatasetName.DNA,
        input_dim=180,
        train_size=1_400,
        validation_size=600,
        test_size=1_186,
        url=f"{_RAW_DATA}/dna.npz",
        sha256="df2d2e7ce1469edfead15e41024e108d64f1b4633cfecfc40ef7107cc22c895c",
    ),
}


PAPER_PRESETS: dict[DatasetName, PaperPreset] = {
    # Table 6's compute-feasible image condition: one 8,000-unit ReLU layer,
    # one fixed shuffled ordering, no direct conditioning weights, and Adagrad.
    DatasetName.BINARIZED_MNIST: PaperPreset(
        dataset=DatasetName.BINARIZED_MNIST,
        hidden_dims=(8_000,),
        activation=ActivationName.RELU,
        optimizer=OptimizerName.ADAGRAD,
        learning_rate=0.01,
        decay=0.0,
        epsilon=1e-6,
        mask_mode=MaskMode.FIXED,
        validation_masks=1,
        test_masks=1,
        batch_size=100,
        patience=30,
        seed=1234,
        direct_input_to_output=False,
        paper_test_nll=88.40,
        paper_test_nll_ci95=0.45,
    ),
    # The official Adult command uses the plain decreasing-learning-rate rule
    # with decrease_constant=0, which is ordinary SGD at a constant 0.01.
    DatasetName.ADULT: PaperPreset(
        dataset=DatasetName.ADULT,
        hidden_dims=(500,),
        activation=ActivationName.TANH,
        optimizer=OptimizerName.SGD,
        learning_rate=0.01,
        decay=0.0,
        epsilon=0.0,
        mask_mode=MaskMode.FIXED,
        validation_masks=1,
        test_masks=1,
        batch_size=100,
        patience=30,
        seed=1234,
        direct_input_to_output=True,
        paper_test_nll=13.12,
        paper_test_nll_ci95=0.05,
    ),
    # The DNA command samples a new full mask after every update and averages
    # 300 masks for validation. The released evaluator averages 1,000 masks.
    DatasetName.DNA: PaperPreset(
        dataset=DatasetName.DNA,
        hidden_dims=(500,),
        activation=ActivationName.RELU,
        optimizer=OptimizerName.ADADELTA,
        learning_rate=1.0,
        decay=0.95,
        epsilon=1e-5,
        mask_mode=MaskMode.EVERY_BATCH,
        validation_masks=300,
        test_masks=1_000,
        batch_size=100,
        patience=30,
        seed=1234,
        direct_input_to_output=True,
        paper_test_nll=79.66,
        paper_test_nll_ci95=0.63,
    ),
}


# The residual candidate retains the original data, objective, optimizer, seed,
# and fixed MADE ordering. Its two degree-preserving blocks offer additional
# nonlinear transformations while reducing the parameter count substantially
# relative to the paper's 8,000-wide single hidden layer.
RESIDUAL_MNIST_PRESET = ImprovementPreset(
    architecture=ArchitectureName.RESIDUAL,
    hidden_dim=2_048,
    residual_blocks=1,
    direct_input_to_output=True,
)


# This candidate retains the paper MADE backbone but adds the masked linear
# autoregressive path omitted by the selected one-hidden-layer paper setting.
# It is an architectural change: output d receives an additional trainable
# function of only x_{<d}, enforced by the same strict degree mask.
DIRECT_MNIST_PRESET = ImprovementPreset(
    architecture=ArchitectureName.DIRECT,
    hidden_dim=8_000,
    residual_blocks=0,
    direct_input_to_output=True,
)


# A deeper candidate retains the successful 8,000-unit path and adds a narrow
# masked nonlinear correction branch. A hidden-to-output skip preserves the
# wide path while the zero-initialized 512-unit branch learns an additional
# causal correction instead of disrupting the established logits at startup.
DEEP_MNIST_HIDDEN_DIMS = (8_000, 512)


# Residual LayerNorm attention-MADE: the original masked autoencoder's order
# and Bernoulli likelihood, with the assignment's residual, normalisation,
# attention, and dropout regularisation applied to the hidden transformation.
# AdamW is the optimiser these blocks are conventionally trained with; it is
# not the architectural claim.
ATTENTION_MNIST_PRESET = ImprovementPreset(
    architecture=ArchitectureName.ATTENTION,
    hidden_dim=256,
    residual_blocks=2,
    direct_input_to_output=False,
    num_heads=4,
    dropout=0.1,
    optimizer=OptimizerName.ADAMW,
    learning_rate=1e-3,
)


# Gated PixelCNN (van den Oord et al., 2016): the vertical and horizontal
# convolution stacks remove the stacked-masked-convolution blind spot, and
# gated tanh-sigmoid units replace ReLU. Same data, Bernoulli likelihood,
# batch size, seed, and checkpoint rule as the reconstruction; AdamW is the
# optimizer this family is conventionally trained with.
GATED_PIXELCNN_MNIST_PRESET = ImprovementPreset(
    architecture=ArchitectureName.GATED_PIXELCNN,
    hidden_dim=96,
    residual_blocks=10,
    direct_input_to_output=False,
    optimizer=OptimizerName.ADAMW,
    learning_rate=1e-3,
)


# Order-agnostic locally masked convolutional MADE (Jain et al., 2020):
# keeps both MADE ideas (masked weights for exact one-pass likelihoods and
# averaging over orderings) but realizes the masks spatially. Training
# cycles eight dihedral S-curve orders (one per batch, exactly MADE's
# mask-resampling protocol) and evaluation averages all eight orders.
LMCONV_ENSEMBLE_MNIST_PRESET = ImprovementPreset(
    architecture=ArchitectureName.LMCONV_ENSEMBLE,
    hidden_dim=64,
    residual_blocks=8,
    direct_input_to_output=False,
    optimizer=OptimizerName.ADAMW,
    learning_rate=1e-3,
)

LMCONV_ENSEMBLE_ORDERS = 8


def paper_preset(dataset: DatasetName | str) -> PaperPreset:
    """Return the immutable paper preset for ``dataset``."""

    return PAPER_PRESETS[DatasetName(dataset)]
