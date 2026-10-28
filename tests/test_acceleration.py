from __future__ import annotations

import numpy as np
import pytest

from lecture2pdf.acceleration.cpu_backend import CPUBackend
from lecture2pdf.acceleration.device import (
    ComputeMode,
    CudaUnavailableError,
    DeviceInfo,
    select_device,
)
from lecture2pdf.acceleration.memory import MemoryBudget
from lecture2pdf.acceleration.profiling import Profiler


def test_cpu_selection_is_explicit_and_truthful():
    selected = select_device("cpu")
    assert selected.mode is ComputeMode.CPU
    assert selected.device == "cpu"
    assert not selected.is_cuda


def test_gpu_selection_does_not_silently_fallback(monkeypatch):
    import lecture2pdf.acceleration.device as device

    monkeypatch.setattr(device, "detect_cuda", lambda: DeviceInfo(False, error="probe failed"))
    with pytest.raises(CudaUnavailableError, match="explicitly requested"):
        select_device("gpu")


def test_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    import lecture2pdf.acceleration.device as device

    monkeypatch.setattr(device, "detect_cuda", lambda: DeviceInfo(False, error="not installed"))
    selected = select_device("auto")
    assert selected.device == "cpu"
    assert selected.info.error == "not installed"


def test_cpu_backend_batch_and_masked_fusion():
    backend = CPUBackend()
    reference = np.zeros((2, 2), dtype=np.uint8)
    candidates = [reference.copy(), np.ones((2, 2), dtype=np.uint8)]
    assert backend.batch_similarity(reference, candidates)[0] == pytest.approx(1.0)
    fused = backend.temporal_fuse(candidates, [np.ones((2, 2), bool), np.zeros((2, 2), bool)])
    np.testing.assert_array_equal(fused, reference)


def test_memory_budget_and_profiler():
    budget = MemoryBudget(device_bytes=10)
    assert budget.batch_size(4, requested=10) == 2
    assert budget.reserve(device_bytes=8)
    assert not budget.reserve(device_bytes=4)
    profiler = Profiler()
    with profiler.stage("unit"):
        pass
    report = profiler.report()[0]
    assert report.name == "unit"
    assert report.calls == 1
