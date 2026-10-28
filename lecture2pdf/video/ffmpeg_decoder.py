"""FFmpeg and OpenCV sequential decoder implementations."""

from __future__ import annotations

import re
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from .decoder import DecodedFrame, DecoderError, VideoDecoder
from .metadata import VideoMetadata

_PTS_RE = re.compile(r"pts_time:([-+]?\d+(?:\.\d+)?)")


class OpenCVDecoder(VideoDecoder):
    backend_name = "opencv"

    def __init__(self, path: Path, *, metadata: VideoMetadata | None = None):
        import cv2
        self.path = path
        self.metadata = metadata
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            self._capture.release()
            raise DecoderError(f"OpenCV could not open video: {path}")

    def __iter__(self) -> Iterator[DecodedFrame]:
        import cv2
        index = 0
        fallback_fps = self.metadata.fps if self.metadata else 30.0
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp <= 0 and index:
                    timestamp = index / fallback_fps
                yield DecodedFrame(frame=frame, timestamp=max(0.0, timestamp), frame_index=index)
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
            self._capture = None


class FFmpegDecoder(VideoDecoder):
    backend_name = "ffmpeg"

    def __init__(self, path: Path, *, metadata: VideoMetadata, hardware_acceleration: bool = False, queue_size: int = 4):
        self.path = path
        self.metadata = metadata
        self.hardware_acceleration = hardware_acceleration
        self.queue_size = max(1, int(queue_size))
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._timestamps: queue.Queue[float] = queue.Queue(maxsize=self.queue_size)
        self._stderr_done = threading.Event()

    def _collect_timestamps(self, pipe) -> None:
        try:
            for raw in iter(pipe.readline, b""):
                match = _PTS_RE.search(raw.decode("utf-8", errors="replace"))
                if match:
                    self._timestamps.put(float(match.group(1)))
        finally:
            self._stderr_done.set()

    def _start(self) -> None:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "info"]
        if self.hardware_acceleration:
            command += ["-hwaccel", "cuda"]
        command += ["-i", str(self.path), "-map", "0:v:0", "-vf", "showinfo", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except OSError as exc:
            raise DecoderError(f"Unable to start FFmpeg: {exc}") from exc
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(target=self._collect_timestamps, args=(self._process.stderr,), daemon=True)
        self._stderr_thread.start()

    def __iter__(self) -> Iterator[DecodedFrame]:
        self._start()
        assert self._process is not None and self._process.stdout is not None
        frame_bytes = self.metadata.width * self.metadata.height * 3
        index = 0
        try:
            while True:
                payload = self._read_exact(self._process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise DecoderError("FFmpeg ended with a partial raw video frame")
                frame = np.frombuffer(payload, dtype=np.uint8).reshape((self.metadata.height, self.metadata.width, 3))
                yield DecodedFrame(frame=frame, timestamp=self._timestamp_for(index), frame_index=index)
                index += 1
            return_code = self._process.wait()
            if return_code != 0:
                raise DecoderError(f"FFmpeg decoder exited with status {return_code}")
        finally:
            self.close()

    def _timestamp_for(self, index: int) -> float:
        try:
            return self._timestamps.get(timeout=2.0)
        except queue.Empty:
            return index / self.metadata.fps

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
