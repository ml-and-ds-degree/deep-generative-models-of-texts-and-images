"""Experiment constants matching Burda et al. (2015).

The controlled comparison changes only the encoder gradient estimator:
architecture, data, initialization, optimizer, particle count, and training
budget must remain identical between IWAE and DReG.
"""

from __future__ import annotations

from enum import StrEnum


class Objective(StrEnum):
    """Supported encoder-gradient estimators.

    DReG is the Stage-2 scientific change. Framework-native checkpointing,
    scheduling, metric accumulation, and accelerator handling are engineering
    modernizations rather than experimental improvements.
    """

    IWAE = "iwae"
    DREG = "dreg"


# Cumulative boundaries of the paper's eight 3**i-pass stages. MultiStepLR is
# an exact framework-native expression of the schedule, not a new schedule.
PAPER_EPOCH_BOUNDARIES: tuple[int, ...] = (1, 4, 13, 40, 121, 364, 1093, 3280)
PAPER_LR_DECAY: float = 10 ** (-1 / 7)
