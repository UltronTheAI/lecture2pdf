"""Bounded host/device memory accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Optional


@dataclass(frozen=True)
class MemorySnapshot:
    process_rss_bytes: Optional[int] = None
    gpu_used_bytes: Optional[int] = None
    gpu_free_bytes: Optional[int] = None
    gpu_total_bytes: Optional[int] = None

    @property
    def process_rss_gib(self) -> Optional[float]:
        return _gib(self.process_rss_bytes)

    @property
    def gpu_used_gib(self) -> Optional[float]:
        return _gib(self.gpu_used_bytes)


def _gib(value: Optional[int]) -> Optional[float]:
    return None if value is None else value / (1024**3)


def get_memory_snapshot() -> MemorySnapshot:
    rss = None
    try:
        psutil = importlib.import_module("psutil")
        rss = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    used = free = total = None
    try:
        cp = importlib.import_module("cupy")
        free, total = (int(v) for v in cp.cuda.runtime.memGetInfo())
        used = total - free
    except Exception:
        pass
    return MemorySnapshot(rss, used, free, total)


class MemoryBudget:
    """Small reservation ledger used to size bounded batches conservatively."""

    def __init__(self, host_bytes: Optional[int] = None, device_bytes: Optional[int] = None, safety_fraction: float = 0.80):
        if not 0 < safety_fraction <= 1:
            raise ValueError("safety_fraction must be in (0, 1]")
        self.host_limit = host_bytes
        self.device_limit = device_bytes
        self.safety_fraction = safety_fraction
        self.host_reserved = 0
        self.device_reserved = 0

    @classmethod
    def from_available_device_memory(cls, safety_fraction: float = 0.80) -> "MemoryBudget":
        snapshot = get_memory_snapshot()
        device = None if snapshot.gpu_free_bytes is None else int(snapshot.gpu_free_bytes * safety_fraction)
        return cls(device_bytes=device, safety_fraction=safety_fraction)

    def reserve(self, host_bytes: int = 0, device_bytes: int = 0) -> bool:
        if host_bytes < 0 or device_bytes < 0:
            raise ValueError("reservation sizes must be non-negative")
        if self.host_limit is not None and self.host_reserved + host_bytes > self.host_limit:
            return False
        if self.device_limit is not None and self.device_reserved + device_bytes > self.device_limit:
            return False
        self.host_reserved += host_bytes
        self.device_reserved += device_bytes
        return True

    def release(self, host_bytes: int = 0, device_bytes: int = 0) -> None:
        self.host_reserved = max(0, self.host_reserved - max(0, host_bytes))
        self.device_reserved = max(0, self.device_reserved - max(0, device_bytes))

    def batch_size(self, item_bytes: int, requested: int = 1, device: bool = True) -> int:
        if item_bytes <= 0 or requested <= 0:
            raise ValueError("item_bytes and requested must be positive")
        limit = self.device_limit if device else self.host_limit
        reserved = self.device_reserved if device else self.host_reserved
        if limit is None:
            return requested
        available = max(0, limit - reserved)
        return min(requested, available // item_bytes)
