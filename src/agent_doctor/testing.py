"""Helpers for testing agents: simulated time and repeatable IDs.

>>> from agent_doctor.testing import simulated_time, deterministic_ids, sleep
>>> with simulated_time(start_ms=1_760_000_000_000), deterministic_ids(seed=7):
...     run_my_agent()          # tools call agent_doctor.testing.sleep(0.5) to "take" 500 ms

Under ``simulated_time`` no real time passes: ``sleep()`` just moves the clock forward,
so recorded durations are exact and the same on every machine.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterator
from contextlib import contextmanager

from agent_doctor import _util


class SimulatedClock:
    def __init__(self, start_ms: int = 1_760_000_000_000):
        self.start_ms = start_ms
        self.seconds = 0.0

    def time_ms(self) -> int:
        return self.start_ms + int(round(self.seconds * 1000))

    def perf(self) -> float:
        return self.seconds

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot sleep a negative time")
        self.seconds += seconds

    advance = sleep

    def set_time_ms(self, epoch_ms: int) -> None:
        """Jump the wall clock to ``epoch_ms`` (never backwards)."""
        target = (epoch_ms - self.start_ms) / 1000
        if target < self.seconds:
            raise ValueError("simulated time cannot go backwards")
        self.seconds = target


@contextmanager
def simulated_time(start_ms: int = 1_760_000_000_000) -> Iterator[SimulatedClock]:
    clock = SimulatedClock(start_ms)
    prev = _util.set_clock(clock)
    try:
        yield clock
    finally:
        _util.set_clock(prev)


@contextmanager
def deterministic_ids(seed: int | str = 0) -> Iterator[None]:
    """Run and span IDs come from a seeded sequence, so reruns produce the same IDs."""
    counter = itertools.count()

    def factory(prefix: str) -> str:
        digest = hashlib.sha256(f"{seed}:{next(counter)}".encode()).hexdigest()
        return f"{prefix}_{digest[:12]}"

    prev = _util.set_id_factory(factory)
    try:
        yield
    finally:
        _util.set_id_factory(prev)


def sleep(seconds: float) -> None:
    """Sleep on the active clock: real time normally, simulated time in tests."""
    _util.get_clock().sleep(seconds)
