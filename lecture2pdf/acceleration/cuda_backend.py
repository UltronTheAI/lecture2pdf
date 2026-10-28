"""CuPy CUDA backend.

CuPy is intentionally imported lazily.  A CUDA backend instance can only be
created from a successful :func:`lecture2pdf.acceleration.device.detect_cuda`
probe, so GPU mode never masquerades as CPU mode.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .device import CudaUnavailableError, DeviceInfo, detect_cuda


class CUDABackend:
    name = "cuda"
    is_cuda = True

    def __init__(self, info: Optional[DeviceInfo] = None):
        info = info or detect_cuda()
        if not info.available or info.provider not in {"cupy", "torch"}:
            raise CudaUnavailableError(
                "CUDABackend requires an operational CuPy or PyTorch CUDA provider. "
                f"Detected: {info.error or info.provider or 'unavailable'}"
            )
        self.info = info
        self.cp = None
        self.torch = None
        if info.provider == "cupy":
            try:
                import cupy as cp
            except Exception as exc:
                raise CudaUnavailableError(f"CuPy could not be imported: {exc}") from exc
            self.cp = cp
            cp.cuda.Device(info.device_index).use()
        else:
            try:
                import torch
            except Exception as exc:
                raise CudaUnavailableError(f"PyTorch could not be imported: {exc}") from exc
            self.torch = torch
            torch.cuda.set_device(info.device_index)

    def array(self, value, dtype=None):
        if self.cp is not None:
            return self.cp.asarray(value, dtype=dtype)
        return self.torch.as_tensor(value, dtype=dtype, device=self._torch_device)

    def to_device(self, value):
        if self.cp is not None:
            return self.cp.asarray(value)
        return self.torch.as_tensor(value, device=self._torch_device)

    def to_host(self, value):
        if self.cp is not None:
            return self.cp.asnumpy(value)
        return value.detach().cpu().numpy()

    def synchronize(self) -> None:
        if self.cp is not None:
            self.cp.cuda.runtime.deviceSynchronize()
        else:
            self.torch.cuda.synchronize(self.info.device_index)

    def grayscale(self, frames):
        if self.cp is None:
            data = self.torch.as_tensor(frames, device=self._torch_device)
            if data.ndim < 3 or data.shape[-1] == 1:
                return data[..., 0] if data.ndim >= 3 and data.shape[-1] == 1 else data
            weights = self.torch.tensor([0.114, 0.587, 0.299], dtype=self.torch.float32, device=self._torch_device)
            return self.torch.tensordot(data[..., :3].to(self.torch.float32), weights, dims=([-1], [0]))
        data = self.cp.asarray(frames)
        if data.ndim < 3 or data.shape[-1] == 1:
            return data[..., 0] if data.ndim >= 3 and data.shape[-1] == 1 else data
        weights = self.cp.asarray([0.114, 0.587, 0.299], dtype=self.cp.float32)
        return self.cp.tensordot(data[..., :3], weights, axes=([-1], [0]))

    def difference(self, left, right) -> float:
        if self.cp is None:
            a = self.torch.as_tensor(left, dtype=self.torch.float32, device=self._torch_device)
            b = self.torch.as_tensor(right, dtype=self.torch.float32, device=self._torch_device)
            if a.shape != b.shape:
                raise ValueError(f"frames must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
            peak = self.torch.maximum(self.torch.max(self.torch.abs(a)), self.torch.max(self.torch.abs(b)))
            scale = self.torch.where(peak > 1.5, self.torch.tensor(255.0, device=self._torch_device), self.torch.tensor(1.0, device=self._torch_device))
            value = self.torch.mean(self.torch.abs(a - b)) / scale
            return float(value.detach().cpu())
        a = self.cp.asarray(left, dtype=self.cp.float32)
        b = self.cp.asarray(right, dtype=self.cp.float32)
        if a.shape != b.shape:
            raise ValueError(f"frames must have the same shape, got {a.shape} and {b.shape}")
        peak = self.cp.maximum(self.cp.max(self.cp.abs(a)), self.cp.max(self.cp.abs(b)))
        scale = self.cp.where(peak > 1.5, self.cp.asarray(255.0), self.cp.asarray(1.0))
        value = self.cp.mean(self.cp.abs(a - b)) / scale
        return float(value.get())

    def batch_similarity(self, reference, candidates: Iterable) -> np.ndarray:
        if self.cp is None:
            ref = self.torch.as_tensor(reference, dtype=self.torch.float32, device=self._torch_device)
            items = list(candidates)
            if not items:
                return np.empty((0,), dtype=np.float32)
            if isinstance(items[0], self.torch.Tensor):
                batch = self.torch.stack([item.to(device=self._torch_device, dtype=self.torch.float32) for item in items])
            else:
                batch = self.torch.as_tensor(np.asarray(items), dtype=self.torch.float32, device=self._torch_device)
            if batch.ndim == ref.ndim:
                batch = batch.unsqueeze(0)
            if tuple(batch.shape[1:]) != tuple(ref.shape):
                raise ValueError("candidate frames must have the same shape as reference")
            distances = self.torch.mean(self.torch.abs(batch - ref), dim=tuple(range(1, batch.ndim)))
            scale = self.torch.clamp(self.torch.max(self.torch.abs(ref)), min=1.0)
            return (1.0 - distances / scale).detach().cpu().numpy().astype(np.float32, copy=False)
        ref = self.cp.asarray(reference, dtype=self.cp.float32)
        items = list(candidates)
        if not items:
            return np.empty((0,), dtype=np.float32)
        batch = self.cp.stack([self.cp.asarray(item, dtype=self.cp.float32) for item in items])
        if batch.ndim == ref.ndim:
            batch = batch[None, ...]
        if batch.shape[1:] != ref.shape:
            raise ValueError("candidate frames must have the same shape as reference")
        distances = self.cp.mean(self.cp.abs(batch - ref), axis=tuple(range(1, batch.ndim)))
        scale = self.cp.maximum(self.cp.max(self.cp.abs(ref)), 1.0)
        return self.cp.asnumpy(1.0 - distances / scale).astype(np.float32, copy=False)

    def temporal_fuse(self, frames: Iterable, masks: Optional[Iterable] = None):
        if self.cp is None:
            stack = self.torch.stack([self.torch.as_tensor(frame, device=self._torch_device) for frame in frames])
            if stack.ndim < 1 or stack.shape[0] == 0:
                raise ValueError("temporal_fuse requires at least one frame")
            if masks is None:
                return self.torch.median(stack, dim=0).values.to(dtype=stack.dtype)
            mask_stack = self.torch.as_tensor(np.asarray(list(masks)), dtype=self.torch.bool, device=self._torch_device)
            expected_mask_shape = (stack.shape[0], *stack.shape[1:3])
            if tuple(mask_stack.shape) != expected_mask_shape:
                raise ValueError("masks must have shape (frames, height, width)")
            condition = mask_stack if stack.ndim == 3 else mask_stack.unsqueeze(-1)
            masked = self.torch.where(condition, stack, self.torch.tensor(float("nan"), device=self._torch_device))
            result = self.torch.nanmedian(masked, dim=0).values
            return self.torch.where(self.torch.isnan(result), stack[0], result).to(dtype=stack.dtype)
        stack = self.cp.asarray(list(frames))
        if stack.ndim < 1 or stack.shape[0] == 0:
            raise ValueError("temporal_fuse requires at least one frame")
        if masks is None:
            return self.cp.median(stack, axis=0).astype(stack.dtype, copy=False)
        mask_stack = self.cp.asarray(list(masks), dtype=bool)
        expected_mask_shape = (stack.shape[0], *stack.shape[1:3])
        if mask_stack.shape != expected_mask_shape:
            raise ValueError("masks must have shape (frames, height, width)")
        expanded = mask_stack if stack.ndim == 3 else mask_stack[..., None]
        masked = self.cp.where(expanded, stack, self.cp.nan)
        result = self.cp.nanmedian(masked, axis=0)
        result = self.cp.where(self.cp.isnan(result), stack[0], result)
        return result.astype(stack.dtype, copy=False)

    @property
    def _torch_device(self):
        return self.torch.device(f"cuda:{self.info.device_index}")
