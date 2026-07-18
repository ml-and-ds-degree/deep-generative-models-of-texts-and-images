"""Experiment constants matching Burda et al. (2015)."""

from __future__ import annotations

from enum import StrEnum


class Objective(StrEnum):
    """Supported encoder-gradient estimators."""

    IWAE = "iwae"
    DREG = "dreg"


PAPER_EPOCH_BOUNDARIES: tuple[int, ...] = (1, 4, 13, 40, 121, 364, 1093, 3280)
PAPER_LR_DECAY: float = 10 ** (-1 / 7)
