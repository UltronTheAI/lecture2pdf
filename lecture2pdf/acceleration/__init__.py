"""Compute backends and device selection for Lecture2PDF.

The package deliberately keeps CUDA libraries optional.  Importing
``lecture2pdf.acceleration`` is therefore safe on CPU-only installations.
"""

from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .device import (
    ComputeMode,
    DeviceInfo,
    DeviceSelection,
    DeviceSelectionError,
    CudaUnavailableError,
    detect_cuda,
    select_device,
)
from .memory import MemoryBudget, MemorySnapshot, get_memory_snapshot
from .profiling import ProfileReport, Profiler, StageTimer


def create_backend(mode: str | ComputeMode = ComputeMode.AUTO):
    """Create the selected backend and return ``(backend, selection)``.

    ``gpu`` remains strict: if the CUDA probe fails, ``select_device`` raises
    before a CPU backend can be created.
    """

    selection = select_device(mode)
    if selection.is_cuda:
        return CUDABackend(selection.info), selection
    return CPUBackend(), selection

__all__ = [
    "CPUBackend",
    "CUDABackend",
    "ComputeMode",
    "DeviceInfo",
    "DeviceSelection",
    "DeviceSelectionError",
    "CudaUnavailableError",
    "detect_cuda",
    "select_device",
    "create_backend",
    "MemoryBudget",
    "MemorySnapshot",
    "get_memory_snapshot",
    "ProfileReport",
    "Profiler",
    "StageTimer",
]
