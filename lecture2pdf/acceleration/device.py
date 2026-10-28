"""Truthful CPU/CUDA capability detection and backend selection.

CUDA is considered available only after a CUDA-capable optional provider can
successfully enumerate a device, allocate a small array, and synchronize.
This prevents a machine with an NVIDIA adapter but a broken Python CUDA
runtime from being reported as a usable GPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
import os
from typing import Any, Optional


class ComputeMode(str, Enum):
    """Requested compute policy."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"

    @classmethod
    def parse(cls, value: str | "ComputeMode") -> "ComputeMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError("compute mode must be one of: auto, cpu, gpu") from exc


@dataclass(frozen=True)
class DeviceInfo:
    """Observed device/runtime capabilities."""

    available: bool
    provider: Optional[str] = None
    name: Optional[str] = None
    device_index: int = 0
    total_memory_bytes: Optional[int] = None
    free_memory_bytes: Optional[int] = None
    runtime_version: Optional[str] = None
    error: Optional[str] = None

    @property
    def total_memory_gib(self) -> Optional[float]:
        return _to_gib(self.total_memory_bytes)

    @property
    def free_memory_gib(self) -> Optional[float]:
        return _to_gib(self.free_memory_bytes)


@dataclass(frozen=True)
class DeviceSelection:
    """Result of resolving a compute policy."""

    mode: ComputeMode
    device: str
    info: DeviceInfo

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def selected(self) -> str:
        """Compatibility name used by the v0.2 orchestration layer."""

        return self.device

    def describe(self) -> str:
        if not self.is_cuda:
            return "CPU"
        provider = self.info.provider or "CUDA"
        name = self.info.name or "CUDA device"
        return f"CUDA ({name}; provider={provider})"


class DeviceSelectionError(RuntimeError):
    """Base class for failures while resolving a compute device."""


class CudaUnavailableError(DeviceSelectionError):
    """Raised when explicit GPU mode cannot be honored."""


def _to_gib(value: Optional[int]) -> Optional[float]:
    return None if value is None else value / (1024**3)


def _probe_cupy() -> DeviceInfo:
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:  # optional dependency; import failures are expected
        return DeviceInfo(False, error=f"CuPy unavailable: {exc}")

    try:
        runtime = cp.cuda.runtime
        count = int(runtime.getDeviceCount())
        if count < 1:
            return DeviceInfo(False, provider="cupy", error="CuPy reports no CUDA devices")
        index = int(os.environ.get("LECTURE2PDF_CUDA_DEVICE", "0"))
        if index < 0 or index >= count:
            return DeviceInfo(False, provider="cupy", error=f"CUDA device index {index} is out of range")
        cp.cuda.Device(index).use()
        props = runtime.getDeviceProperties(index)
        name = props.get("name", "CUDA device") if isinstance(props, dict) else "CUDA device"
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free, total = runtime.memGetInfo()
        probe = cp.zeros((1,), dtype=cp.float32)
        probe += 1
        runtime.deviceSynchronize()
        del probe
        version = getattr(runtime, "runtimeGetVersion", lambda: None)()
        return DeviceInfo(
            True,
            provider="cupy",
            name=str(name),
            device_index=index,
            total_memory_bytes=int(total),
            free_memory_bytes=int(free),
            runtime_version=str(version) if version is not None else None,
        )
    except Exception as exc:
        return DeviceInfo(False, provider="cupy", error=f"CuPy CUDA probe failed: {exc}")


def _probe_torch() -> DeviceInfo:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # optional dependency
        return DeviceInfo(False, error=f"PyTorch unavailable: {exc}")

    try:
        if not bool(torch.cuda.is_available()):
            return DeviceInfo(False, provider="torch", error="PyTorch reports CUDA unavailable")
        index = int(os.environ.get("LECTURE2PDF_CUDA_DEVICE", "0"))
        count = int(torch.cuda.device_count())
        if index < 0 or index >= count:
            return DeviceInfo(False, provider="torch", error=f"CUDA device index {index} is out of range")
        torch.cuda.set_device(index)
        probe = torch.zeros((1,), device=f"cuda:{index}")
        probe.add_(1)
        torch.cuda.synchronize(index)
        free, total = torch.cuda.mem_get_info(index)
        return DeviceInfo(
            True,
            provider="torch",
            name=str(torch.cuda.get_device_name(index)),
            device_index=index,
            total_memory_bytes=int(total),
            free_memory_bytes=int(free),
            runtime_version=getattr(torch.version, "cuda", None),
        )
    except Exception as exc:
        return DeviceInfo(False, provider="torch", error=f"PyTorch CUDA probe failed: {exc}")


def detect_cuda() -> DeviceInfo:
    """Return the first operational CUDA provider, or a diagnostic result."""

    probes = (_probe_cupy, _probe_torch)
    failures: list[str] = []
    for probe in probes:
        result = probe()
        if result.available:
            return result
        if result.error:
            failures.append(result.error)
    return DeviceInfo(False, error="; ".join(failures) or "No CUDA provider is installed")


def select_device(mode: str | ComputeMode = ComputeMode.AUTO) -> DeviceSelection:
    """Resolve ``auto``, ``cpu`` or ``gpu`` without silent GPU fallback."""

    requested = ComputeMode.parse(mode)
    if requested is ComputeMode.CPU:
        return DeviceSelection(requested, "cpu", DeviceInfo(True, provider="numpy", name="CPU"))

    cuda = detect_cuda()
    if requested is ComputeMode.GPU and not cuda.available:
        diagnostic = cuda.error or "no operational CUDA provider was detected"
        raise CudaUnavailableError(
            "GPU compute was explicitly requested, but CUDA is unavailable. "
            f"{diagnostic}. Install a compatible CUDA-enabled CuPy or PyTorch build, "
            "verify the NVIDIA driver, or use --compute cpu/auto."
        )
    if cuda.available:
        return DeviceSelection(requested, "cuda", cuda)
    return DeviceSelection(requested, "cpu", DeviceInfo(True, provider="numpy", name="CPU", error=cuda.error))


def provider_module(info: DeviceInfo) -> Any:
    """Import the selected provider; useful to backend factories."""

    if info.provider == "cupy":
        return importlib.import_module("cupy")
    if info.provider == "torch":
        return importlib.import_module("torch")
    return None
