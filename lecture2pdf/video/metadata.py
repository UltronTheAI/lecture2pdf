"""Video metadata discovery and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction
from typing import Any

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int | None = None
    codec: str | None = None
    pixel_format: str | None = None
    time_base: str | None = None
    variable_frame_rate: bool = False
    container: str | None = None
    source: str = "unknown"

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def _ratio(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _ffprobe(path: Path) -> VideoMetadata | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "format=format_name,duration:stream=width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload: dict[str, Any] = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
        fmt = payload.get("format") or {}
        fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
        raw_count = stream.get("nb_frames")
        frame_count = int(raw_count) if str(raw_count or "").isdigit() else None
        nominal = _ratio(stream.get("r_frame_rate"))
        variable = bool(fps and nominal and abs(fps - nominal) > 1e-3)
        return VideoMetadata(
            path=path, width=int(stream.get("width") or 0), height=int(stream.get("height") or 0),
            fps=fps, duration=duration, frame_count=frame_count, codec=stream.get("codec_name"),
            pixel_format=stream.get("pix_fmt"), time_base=stream.get("time_base"),
            variable_frame_rate=variable, container=fmt.get("format_name"), source="ffprobe",
        )
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _opencv(path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        duration = frame_count / fps if frame_count and fps > 0 else 0.0
        return VideoMetadata(
            path=path, width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), fps=fps,
            duration=duration, frame_count=frame_count, source="opencv",
        )
    finally:
        capture.release()


def validate_metadata(metadata: VideoMetadata) -> VideoMetadata:
    if not metadata.path.is_file():
        raise FileNotFoundError(metadata.path)
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError(f"Video has invalid dimensions: {metadata.width}x{metadata.height}")
    if metadata.fps <= 0:
        raise ValueError(f"Video has no usable frame rate: {metadata.path}")
    if metadata.duration < 0:
        raise ValueError(f"Video has invalid duration: {metadata.duration}")
    return metadata


def probe_video(path: str | Path, *, require_supported_extension: bool = False) -> VideoMetadata:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if require_supported_extension and source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported video extension {source.suffix!r}")
    return validate_metadata(_ffprobe(source) or _opencv(source))
