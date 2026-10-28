"""NumPy reference backend used by CPU mode and as a correctness baseline."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


class CPUBackend:
    name = "cpu"
    is_cuda = False

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def to_device(self, value):
        return np.asarray(value)

    def to_host(self, value):
        return np.asarray(value)

    def synchronize(self) -> None:
        return None

    def grayscale(self, frames: np.ndarray) -> np.ndarray:
        data = np.asarray(frames)
        if data.ndim < 3 or data.shape[-1] == 1:
            return data[..., 0] if data.ndim >= 3 and data.shape[-1] == 1 else data
        # OpenCV-style BGR weights; also sensible for RGB content.
        weights = np.asarray([0.114, 0.587, 0.299], dtype=np.float32)
        return np.tensordot(data[..., :3], weights, axes=([-1], [0]))

    def difference(self, left: np.ndarray, right: np.ndarray) -> float:
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError(f"frames must have the same shape, got {a.shape} and {b.shape}")
        peak = max(float(np.max(np.abs(a))) if a.size else 0.0, float(np.max(np.abs(b))) if b.size else 0.0, 1.0)
        scale = 255.0 if peak > 1.5 else 1.0
        return float(np.mean(np.abs(a - b)) / scale)

    def batch_similarity(self, reference, candidates: Iterable) -> np.ndarray:
        ref = np.asarray(reference, dtype=np.float32)
        values = [self.difference(ref, candidate) for candidate in candidates]
        return 1.0 - np.asarray(values, dtype=np.float32)

    def temporal_fuse(self, frames: Iterable[np.ndarray], masks: Optional[Iterable[np.ndarray]] = None) -> np.ndarray:
        stack = np.asarray(list(frames))
        if stack.ndim < 1 or stack.shape[0] == 0:
            raise ValueError("temporal_fuse requires at least one frame")
        if masks is None:
            return np.median(stack, axis=0).astype(stack.dtype, copy=False)
        mask_stack = np.asarray(list(masks), dtype=bool)
        expected_mask_shape = (stack.shape[0], *stack.shape[1:3])
        if mask_stack.shape != expected_mask_shape:
            raise ValueError("masks must have shape (frames, height, width)")
        expanded = mask_stack if stack.ndim == 3 else mask_stack[..., None]
        masked = np.where(expanded, stack, np.nan)
        result = np.nanmedian(masked, axis=0)
        fallback = stack[0]
        result = np.where(np.isnan(result), fallback, result)
        return result.astype(stack.dtype, copy=False)
