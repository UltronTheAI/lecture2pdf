"""Decoder interfaces and runtime backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterator

from .metadata import VideoMetadata, probe_video


class DecoderError(RuntimeError):
    """Raised when a selected decoder cannot start or decode reliably."""


@dataclass
class DecodedFrame:
    frame: object
    timestamp: float
    frame_index: int
    duration: float | None = None


@dataclass(frozen=True)
class DecoderSelection:
    """Resolved decoder name used by the v0.2 orchestration layer."""

    selected: str
    hardware_acceleration: bool = False
    reason: str = ""


class VideoDecoder:
    metadata: VideoMetadata
    backend_name = "unknown"
    hardware_acceleration = False

    def __iter__(self) -> Iterator[DecodedFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def ffmpeg_capabilities() -> dict[str, object]:
    """Return feature-detected FFmpeg capabilities; never assume NVDEC."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"available": False, "nvdec": False, "decoders": (), "hwaccels": ()}
    try:
        decoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-decoders"], capture_output=True, text=True, check=False
        ).stdout
        hwaccels = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return {"available": False, "nvdec": False, "decoders": (), "hwaccels": ()}
    names = tuple(name for name in ("h264_cuvid", "hevc_cuvid", "av1_cuvid", "vp9_cuvid") if name in decoders)
    accel_names = tuple(line.strip() for line in hwaccels.splitlines() if line.strip() and not line.startswith("Hardware"))
    return {"available": True, "nvdec": bool(names and "cuda" in hwaccels.lower()), "decoders": names, "hwaccels": accel_names}


def open_decoder(
    path: str | Path,
    *,
    backend: str = "auto",
    hardware: str = "auto",
    queue_size: int = 4,
) -> VideoDecoder:
    """Create a decoder after validating the input and detecting capabilities."""
    backend = backend.lower()
    hardware = hardware.lower()
    if backend not in {"auto", "ffmpeg", "opencv", "nvdec"}:
        raise ValueError("decoder must be one of: auto, ffmpeg, opencv, nvdec")
    if hardware not in {"auto", "on", "off"}:
        raise ValueError("decoder hardware mode must be one of: auto, on, off")
    source = Path(path).expanduser()
    metadata = probe_video(source)
    capabilities = ffmpeg_capabilities()
    if backend == "auto":
        backend = "ffmpeg" if capabilities["available"] else "opencv"
    if backend == "nvdec":
        backend = "ffmpeg"
        hardware = "on"
    if backend == "ffmpeg" and not capabilities["available"]:
        raise DecoderError("FFmpeg decoder requested but ffmpeg.exe was not found on PATH")
    # ``ffmpeg -decoders`` only proves that the binary was built with CUVID
    # support. It does not prove that a compatible driver/GPU is operational.
    # Keep auto deterministic and software-backed; callers can request `on`
    # and receive a clear runtime failure if the hardware path is unusable.
    use_hw = hardware == "on"
    if use_hw and not bool(capabilities.get("nvdec")):
        if hardware == "on":
            raise DecoderError("Hardware decoding requested, but FFmpeg reports no usable NVDEC/CUDA decoder")
        use_hw = False
    if backend == "ffmpeg":
        from .ffmpeg_decoder import FFmpegDecoder
        return FFmpegDecoder(source, metadata=metadata, hardware_acceleration=use_hw, queue_size=queue_size)
    from .ffmpeg_decoder import OpenCVDecoder
    return OpenCVDecoder(source, metadata=metadata)


def choose_decoder(path: str | Path, requested: str = "auto", compute: str = "cpu") -> DecoderSelection:
    """Resolve a CLI decoder request using observed FFmpeg capabilities.

    ``nvdec`` is never inferred from an NVIDIA adapter alone. It is returned
    only when FFmpeg reports both CUDA hwaccel support and a CUVID decoder.
    ``auto`` deliberately selects software FFmpeg; the caller can request
    ``nvdec`` explicitly when hardware decode is desired.
    """
    requested = str(requested).lower()
    if requested not in {"auto", "opencv", "ffmpeg", "nvdec"}:
        raise ValueError("decoder must be one of: auto, opencv, ffmpeg, nvdec")
    capabilities = ffmpeg_capabilities()
    if requested == "opencv":
        return DecoderSelection("opencv", reason="explicit OpenCV selection")
    if requested == "ffmpeg":
        if not capabilities["available"]:
            raise DecoderError("FFmpeg decoder requested but ffmpeg.exe was not found on PATH")
        return DecoderSelection("ffmpeg", reason="explicit FFmpeg selection")
    if requested == "nvdec":
        if not capabilities["available"] or not capabilities["nvdec"]:
            raise DecoderError("NVDEC requested, but FFmpeg reports no usable CUDA/CUVID decoder")
        return DecoderSelection("nvdec", hardware_acceleration=True, reason="FFmpeg CUVID/CUDA support detected")
    if capabilities["available"]:
        return DecoderSelection("ffmpeg", reason="FFmpeg detected; using bounded sequential software decode")
    return DecoderSelection("opencv", reason="FFmpeg unavailable; using OpenCV fallback")
