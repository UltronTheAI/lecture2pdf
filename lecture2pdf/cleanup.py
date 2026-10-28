"""Safe cleanup of Lecture2PDF-generated artifacts.

The cleanup routine never deletes source videos or arbitrary user files. It
only removes known cache/build/generated-artifact names beneath the selected
project root. Use ``remove_outputs=True`` to remove generated ``*_assets``
directories as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class CleanupReport:
    removed: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def removed_count(self) -> int:
        return len(self.removed)


_CACHE_DIR_NAMES = {
    ".lecture2pdf-cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", "build", "dist", "benchmark_outputs",
}
_CACHE_FILE_SUFFIXES = {".pyc", ".pyo", ".sqlite", ".sqlite-shm", ".sqlite-wal"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def cleanup_generated(root: str | Path = ".", *, remove_outputs: bool = False) -> CleanupReport:
    """Remove known generated caches beneath ``root``.

    By default this removes caches, Python artifacts, package build output,
    and benchmark output. With ``remove_outputs=True`` it also removes
    ``*_assets`` directories and generated log/temp files. Source videos and
    PDFs are intentionally preserved.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    targets: list[Path] = []
    for path in root_path.rglob("*"):
        if not _is_within(path, root_path):
            continue
        if path.is_dir():
            if path.name in _CACHE_DIR_NAMES or (remove_outputs and path.name.endswith("_assets")):
                targets.append(path)
        elif path.is_file():
            if path.suffix.lower() in _CACHE_FILE_SUFFIXES or (remove_outputs and path.suffix.lower() in {".log", ".tmp"}):
                targets.append(path)
    # Remove children before parents and de-duplicate nested targets.
    unique = sorted({p.resolve() for p in targets}, key=lambda p: len(p.parts), reverse=True)
    selected: list[Path] = []
    for path in unique:
        if not any(_is_within(path, parent) for parent in selected):
            selected.append(path)
    removed: list[str] = []
    for path in selected:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(str(path))
    return CleanupReport(tuple(removed), ())


def cleanup_all(root: str | Path = ".") -> CleanupReport:
    """Explicit alias for cleaning all supported generated artifacts."""
    return cleanup_generated(root, remove_outputs=True)
