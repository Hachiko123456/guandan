# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/intervals.py.
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen = True, slots = True)
class IntervalTimer:
    """Interval scheduler helper for periodic/final-triggered actions.

    Interval semantics:
    - ``interval < 0``: disabled.
    - ``interval == 0``: final step only.
    - ``interval > 0``: periodic steps plus the final step.
    """

    interval: int
    total_steps: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "interval", int(self.interval))
        if self.total_steps is not None:
            object.__setattr__(self, "total_steps", int(self.total_steps))

    @property
    def active(self) -> bool:
        return self.interval >= 0

    def triggered(self, step: int, *, previous_step: int | None = None) -> bool:
        resolved_step = int(step)
        if resolved_step <= 0 or not self.active:
            return False
        if previous_step is None:
            return self._triggered_from_schedule(resolved_step)
        return self._triggered_from_progress(resolved_step, previous_step = int(previous_step))

    def _triggered_from_schedule(self, step: int) -> bool:
        if self.total_steps is None:
            raise ValueError("IntervalTimer.triggered() requires total_steps on the timer for schedule checks.")
        if self.total_steps <= 0:
            return False
        if step == self.total_steps:
            return True
        return self._crossed_periodic_boundary(step, previous_step = step - 1)

    def _triggered_from_progress(self, step: int, *, previous_step: int) -> bool:
        if step <= previous_step:
            return False
        if self.total_steps is not None and previous_step < self.total_steps <= step:
            return True
        return self._crossed_periodic_boundary(step, previous_step = previous_step)

    def _crossed_periodic_boundary(self, step: int, *, previous_step: int) -> bool:
        if self.interval <= 0:
            return False
        return (step // self.interval) > (previous_step // self.interval)


__all__ = ["IntervalTimer"]
