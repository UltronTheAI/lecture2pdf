"""Deterministic fingerprints for inputs and processing configurations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def canonicalize(value: Any) -> Any:
    """Return a JSON-compatible, deterministic representation of *value*."""

    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted((canonicalize(item) for item in value), key=repr)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonicalize(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return canonicalize({name: getattr(value, name) for name in value.__dataclass_fields__})
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def fingerprint_config(config: Mapping[str, Any] | Any, *, version: str = "1") -> str:
    payload = {"fingerprint_version": version, "config": canonicalize(config)}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_file(
    path: str | os.PathLike[str], *, sample_bytes: int = 1024 * 1024
) -> str:
    """Fingerprint file identity plus head/tail content without loading it all."""

    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    digest.update(str(source.resolve()).encode("utf-8", "surrogateescape"))
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with source.open("rb") as handle:
        head = handle.read(sample_bytes)
        digest.update(head)
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def job_fingerprint(
    video_path: str | os.PathLike[str], config: Mapping[str, Any] | Any, *, version: str = "1"
) -> str:
    payload = {
        "fingerprint_version": version,
        "video": fingerprint_file(video_path),
        "config": fingerprint_config(config, version=version),
    }
    return fingerprint_config(payload, version=version)
