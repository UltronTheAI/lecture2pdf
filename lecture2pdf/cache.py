"""Bounded, deterministic on-disk layout for resumable Lecture2PDF jobs."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .atomic import atomic_write_bytes, atomic_write_json


@dataclass(frozen=True)
class CacheLayout:
    root: Path
    job_id: str

    @property
    def job_dir(self) -> Path:
        return self.root / "jobs" / self.job_id

    @property
    def database(self) -> Path:
        return self.job_dir / "metadata.sqlite"

    @property
    def frames(self) -> Path:
        return self.job_dir / "frames"

    @property
    def slides(self) -> Path:
        return self.job_dir / "slides"

    @property
    def reconstructed(self) -> Path:
        return self.job_dir / "reconstructed"

    @property
    def ocr(self) -> Path:
        return self.job_dir / "ocr"

    @property
    def checkpoints(self) -> Path:
        return self.job_dir / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.job_dir / "logs"

    def create(self) -> "CacheLayout":
        for directory in (
            self.job_dir,
            self.frames,
            self.slides,
            self.reconstructed,
            self.ocr,
            self.checkpoints,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def child(self, category: str, name: str) -> Path:
        directory = getattr(self, category)
        if not isinstance(directory, Path):
            raise ValueError(f"Unknown cache category: {category}")
        candidate = (directory / name).resolve()
        if os.path.commonpath((str(directory.resolve()), str(candidate))) != str(directory.resolve()):
            raise ValueError("Cache path escapes its category directory")
        return candidate

    def write_bytes(self, category: str, name: str, data: bytes) -> Path:
        return atomic_write_bytes(self.child(category, name), data)

    def write_json(self, category: str, name: str, value: object) -> Path:
        return atomic_write_json(self.child(category, name), value)

    def clear(self) -> None:
        """Remove this job's cache only; the parent cache remains intact."""

        if self.job_dir.exists():
            shutil.rmtree(self.job_dir)


def make_cache_layout(cache_dir: str | os.PathLike[str], job_id: str) -> CacheLayout:
    if not job_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in job_id):
        raise ValueError("job_id must contain only letters, digits, '-' or '_'")
    return CacheLayout(Path(cache_dir), job_id).create()
