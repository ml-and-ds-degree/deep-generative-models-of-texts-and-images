"""Experiment constants matching Burda et al. (2015).

The controlled comparison changes only the encoder gradient estimator:
architecture, data, initialization, optimizer, particle count, and training
budget must remain identical between IWAE and DReG.
"""

from __future__ import annotations

from enum import StrEnum

from attrs import define, field, validators


class Objective(StrEnum):
    """Supported encoder-gradient estimators.

    DReG is the Stage-2 scientific change. Framework-native checkpointing,
    scheduling, metric accumulation, and accelerator handling are engineering
    modernizations rather than experimental improvements.
    """

    IWAE = "iwae"
    DREG = "dreg"


class LearningRateSchedule(StrEnum):
    """Learning-rate schedules for faithful and accelerated experiments."""

    PAPER = "paper"
    COSINE = "cosine"


@define(frozen=True)
class ProgressiveParticleSchedule:
    """Validated progressive-particle curriculum for a finite training budget."""

    max_epochs: int = field(
        validator=validators.and_(validators.instance_of(int), validators.gt(0))
    )
    warmup_particles: int = field(
        validator=validators.and_(validators.instance_of(int), validators.gt(0))
    )
    middle_particles: int = field(
        validator=validators.and_(validators.instance_of(int), validators.gt(0))
    )
    final_particles: int = field(
        validator=validators.and_(validators.instance_of(int), validators.gt(0))
    )

    @final_particles.validator
    def _particle_counts_are_non_decreasing(self, _attribute: object, _value: int) -> None:
        if not self.warmup_particles <= self.middle_particles <= self.final_particles:
            raise ValueError("particle counts must be non-decreasing")

    def stages(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return training particle counts and zero-based transition boundaries."""
        if self.max_epochs < 6 or self.warmup_particles == self.final_particles:
            return (self.final_particles,), ()

        first_boundary = self.max_epochs // 2
        second_boundary = 5 * self.max_epochs // 6
        counts = (self.warmup_particles, self.middle_particles, self.final_particles)
        if self.warmup_particles == self.middle_particles:
            return (self.warmup_particles, self.final_particles), (second_boundary,)
        if self.middle_particles == self.final_particles:
            return (self.warmup_particles, self.final_particles), (first_boundary,)
        return counts, (first_boundary, second_boundary)


# Cumulative boundaries of the paper's eight 3**i-pass stages. MultiStepLR is
# an exact framework-native expression of the schedule, not a new schedule.
PAPER_EPOCH_BOUNDARIES: tuple[int, ...] = (1, 4, 13, 40, 121, 364, 1093, 3280)
PAPER_LR_DECAY: float = 10 ** (-1 / 7)
