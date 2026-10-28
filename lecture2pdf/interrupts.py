"""Cooperative Ctrl+C handling for decoders and checkpointed pipeline stages."""

from __future__ import annotations

import signal
from types import FrameType
from typing import Callable


class InterruptedRun(KeyboardInterrupt):
    """Raised at a safe pipeline boundary after an interruption request."""


class InterruptController:
    def __init__(self, on_interrupt: Callable[[], None] | None = None):
        self.requested = False
        self._on_interrupt = on_interrupt
        self._previous: dict[int, object] = {}

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True
        if self._on_interrupt:
            self._on_interrupt()

    def __enter__(self) -> "InterruptController":
        signals = {signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)}
        for signum in signals:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def raise_if_requested(self) -> None:
        if self.requested:
            raise InterruptedRun("Processing interrupted; the latest checkpoint is available for resume.")
