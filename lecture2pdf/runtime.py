from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import os
import sqlite3
import tempfile
import time

from .acceleration.memory import get_memory_snapshot

LOG = logging.getLogger(__name__)


def fingerprint(path: Path, config: dict) -> str:
    stat = path.stat()
    payload = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "config": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def atomic_write_json(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    except Exception:
        try:
            os.unlink(raw)
        except OSError:
            pass
        raise


@dataclass
class CacheLayout:
    root: Path

    @property
    def db(self): return self.root / "metadata.sqlite"
    @property
    def frames(self): return self.root / "frames"
    @property
    def slides(self): return self.root / "slides"
    @property
    def reconstructed(self): return self.root / "reconstructed"
    @property
    def ocr(self): return self.root / "ocr"
    @property
    def checkpoints(self): return self.root / "checkpoints"

    def create(self):
        for path in (self.frames, self.slides, self.reconstructed, self.ocr, self.checkpoints):
            path.mkdir(parents=True, exist_ok=True)
        return self


class CheckpointStore:
    def __init__(self, layout: CacheLayout, job_id: str, config_fingerprint: str):
        self.layout = layout.create()
        self.job_id = job_id
        self.config_fingerprint = config_fingerprint
        self.connection = sqlite3.connect(self.layout.db)
        self.connection.execute("CREATE TABLE IF NOT EXISTS job (job_id TEXT PRIMARY KEY, config_fingerprint TEXT NOT NULL, updated REAL NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS checkpoint (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL)")
        self.connection.execute("INSERT OR REPLACE INTO job(job_id, config_fingerprint, updated) VALUES (?,?,?)", (job_id, config_fingerprint, time.time()))
        self.connection.commit()

    def get(self, key, default=None):
        row = self.connection.execute("SELECT value FROM checkpoint WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        self.connection.execute("INSERT OR REPLACE INTO checkpoint(key,value,updated) VALUES (?,?,?)", (key, encoded, time.time()))
        self.connection.commit()

    def close(self):
        self.connection.close()


class ProgressTracker:
    def __init__(self, total_seconds=0.0, verbose=False, quiet=False, log_file=None, device=None):
        self.total = max(0.0, float(total_seconds))
        self.verbose = verbose
        self.quiet = quiet
        self.device = device
        self.started = time.perf_counter()
        self.last_emit = 0.0
        self.count = 0
        self.slide_count = 0
        self.duplicate_count = 0
        self.log_file = open(log_file, "a", encoding="utf-8") if log_file else None

    def update(self, stage, timestamp=0.0, count=None, slides=None, duplicates=None, force=False):
        self.last_timestamp = float(timestamp)
        self.count = self.count if count is None else count
        self.slide_count = self.slide_count if slides is None else slides
        self.duplicate_count = self.duplicate_count if duplicates is None else duplicates
        now = time.perf_counter()
        if not force and now - self.last_emit < 0.5:
            return
        self.last_emit = now
        elapsed = max(1e-6, now - self.started)
        rate = self.count / elapsed
        remaining = max(0.0, self.total - timestamp)
        eta = remaining / max(rate, 1e-6) if self.total else 0.0
        snapshot = get_memory_snapshot() if self.device else None
        allocated = snapshot.gpu_used_bytes if snapshot and getattr(self.device, "is_cuda", False) else None
        percent = (timestamp / self.total * 100.0) if self.total else 0.0
        line = (f"Stage: {stage} | Processed: {timestamp:7.1f}s/{self.total:7.1f}s | "
                f"Progress: {percent:5.1f}% | Sample FPS: {rate:6.2f} | "
                f"Unique slides: {self.slide_count} | Duplicates: {self.duplicate_count} | ETA: {eta:6.1f}s")
        if allocated is not None:
            line += f" | VRAM: {allocated / 1024**3:.2f} GB"
        if snapshot and snapshot.process_rss_bytes is not None:
            line += f" | RAM: {snapshot.process_rss_bytes / 1024**3:.2f} GB"
        if not self.quiet:
            print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()
