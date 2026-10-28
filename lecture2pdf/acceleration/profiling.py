"""Low-overhead stage timing with correct CUDA synchronization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class ProfileReport:
    name: str
    calls: int
    total_seconds: float
    average_seconds: float


class StageTimer:
    def __init__(self, name: str, synchronize: Optional[Callable[[], None]] = None):
        self.name = name
        self.synchronize = synchronize
        self.elapsed_seconds = 0.0
        self._started = 0.0

    def __enter__(self) -> "StageTimer":
        if self.synchronize:
            self.synchronize()
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.synchronize:
            self.synchronize()
        self.elapsed_seconds = time.perf_counter() - self._started


class Profiler:
    def __init__(self, synchronize: Optional[Callable[[], None]] = None):
        self.synchronize = synchronize
        self._totals = defaultdict(float)
        self._calls = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTimer]:
        with StageTimer(name, self.synchronize) as timer:
            yield timer
        self._totals[name] += timer.elapsed_seconds
        self._calls[name] += 1

    def report(self) -> list[ProfileReport]:
        return [
            ProfileReport(name, self._calls[name], total, total / self._calls[name])
            for name, total in sorted(self._totals.items())
        ]

    def reset(self) -> None:
        self._totals.clear()
        self._calls.clear()
