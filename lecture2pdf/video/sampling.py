"""Bounded, timestamp-aware sampling for decoded frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol


@dataclass
class Sample:
    """A sampled frame, retaining its presentation timestamp in seconds."""

    time: float
    frame: object
    frame_index: int | None = None


class TimestampedFrame(Protocol):
    timestamp: float
    frame: object
    frame_index: int


def _crop_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    height, width = frame.shape[:2]
    x0 = max(0, min(width - 1, round(x * width)))
    y0 = max(0, min(height - 1, round(y * height)))
    x1 = max(x0 + 1, min(width, round((x + w) * width)))
    y1 = max(y0 + 1, min(height, round((y + h) * height)))
    return frame[y0:y1, x0:x1]


def sample_frames(
    frames: Iterable[TimestampedFrame],
    fps: float = 2.0,
    roi=None,
    start_time: float = 0.0,
) -> Iterator[Sample]:
    """Select approximately ``fps`` samples from a sequential frame stream.

    Selection is based on presentation timestamps rather than frame number,
    so variable-frame-rate streams are handled without loading the video into
    memory. Only the current decoded frame is retained by this generator.
    """
    if fps <= 0:
        raise ValueError("sample fps must be positive")
    interval = 1.0 / fps
    next_time = max(0.0, float(start_time))
    for item in frames:
        timestamp = max(0.0, float(item.timestamp))
        if timestamp + 1e-6 < next_time:
            continue
        frame = _crop_roi(item.frame, roi)
        if getattr(frame, "size", 0):
            yield Sample(timestamp, frame, getattr(item, "frame_index", None))
        next_time = timestamp + interval


def sample_video(path, fps: float = 2.0, roi=None, decoder: str = "auto", start_time: float = 0.0) -> Iterator[Sample]:
    """Backward-compatible convenience wrapper around the decoder adapter."""
    from .decoder import open_decoder

    with open_decoder(path, backend=decoder) as stream:
        yield from sample_frames(stream, fps=fps, roi=roi, start_time=start_time)
