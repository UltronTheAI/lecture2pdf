"""Low-noise progress metrics and reporting for long-running conversions."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def resource_usage() -> tuple[int | None, int | None]:
    """Return process RAM and allocated CUDA memory when optional providers exist."""

    ram_bytes: int | None = None
    vram_bytes: int | None = None
    try:
        import psutil  # type: ignore

        ram_bytes = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            vram_bytes = int(torch.cuda.memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return ram_bytes, vram_bytes


@dataclass
class ProgressMetrics:
    total_duration: float | None = None
    processed_timestamp: float = 0.0
    sampled_frames: int = 0
    unique_slides: int = 0
    duplicates_rejected: int = 0
    stage: str = "starting"
    started_at: float = field(default_factory=time.monotonic)
    last_update: float = field(default_factory=time.monotonic)

    def update(self, *, processed_timestamp: float | None = None, **values: Any) -> None:
        if processed_timestamp is not None:
            self.processed_timestamp = max(self.processed_timestamp, processed_timestamp)
        for key, value in values.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown progress metric: {key}")
            setattr(self, key, value)
        self.last_update = time.monotonic()

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def processing_fps(self) -> float:
        return self.processed_timestamp / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if not self.total_duration or self.processing_fps <= 0:
            return None
        return max(0.0, (self.total_duration - self.processed_timestamp) / self.processing_fps)

    def snapshot(self, *, ram_bytes: int | None = None, vram_bytes: int | None = None) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            elapsed=self.elapsed,
            processing_fps=self.processing_fps,
            eta_seconds=self.eta_seconds,
            ram_bytes=ram_bytes,
            vram_bytes=vram_bytes,
        )
        return data


class ProgressReporter:
    def __init__(
        self,
        *,
        metrics: ProgressMetrics | None = None,
        stream: TextIO | None = None,
        quiet: bool = False,
        verbose: bool = False,
        log_file: str | os.PathLike[str] | None = None,
        interval_seconds: float = 1.0,
    ):
        self.metrics = metrics or ProgressMetrics()
        self.stream = stream
        self.quiet = quiet
        self.verbose = verbose
        self.interval_seconds = interval_seconds
        self._last_emit = 0.0
        self._log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
        self._logger = logging.getLogger("lecture2pdf")

    def emit(self, *, force: bool = False, ram_bytes: int | None = None, vram_bytes: int | None = None) -> str | None:
        now = time.monotonic()
        if not force and now - self._last_emit < self.interval_seconds:
            return None
        self._last_emit = now
        snapshot = self.metrics.snapshot(ram_bytes=ram_bytes, vram_bytes=vram_bytes)
        total = snapshot["total_duration"]
        progress = (snapshot["processed_timestamp"] / total * 100) if total else 0.0
        message = (
            f"Stage: {snapshot['stage']} | "
            f"Processed: {format_duration(snapshot['processed_timestamp'])}"
            f" / {format_duration(total)} ({progress:.1f}%) | "
            f"{snapshot['processing_fps']:.2f} video sec/s | "
            f"Slides: {snapshot['unique_slides']} | "
            f"Duplicates: {snapshot['duplicates_rejected']} | "
            f"ETA: {format_duration(snapshot['eta_seconds'])}"
        )
        if self._log_handle:
            self._log_handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            self._log_handle.flush()
        if self.verbose:
            self._logger.debug("%s", message)
        if not self.quiet and self.stream is not None:
            self.stream.write(message + "\n")
            self.stream.flush()
        return message

    def close(self) -> None:
        if self._log_handle:
            self._log_handle.close()

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
