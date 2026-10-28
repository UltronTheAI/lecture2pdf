"""Job-level wiring for cache, fingerprints, progress, and interruption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .cache import CacheLayout, make_cache_layout
from .checkpoint import CheckpointStore
from .fingerprints import fingerprint_config, fingerprint_file, job_fingerprint
from .interrupts import InterruptController
from .progress import ProgressMetrics, ProgressReporter


@dataclass
class JobContext:
    """Resources shared by decode, detection, reconstruction, OCR, and PDF stages."""

    video_path: Path
    job_id: str
    video_fingerprint: str
    config_fingerprint: str
    cache: CacheLayout
    checkpoints: CheckpointStore
    progress: ProgressReporter
    interrupts: InterruptController

    @classmethod
    def open(
        cls,
        video_path: str | Path,
        config: Mapping[str, Any] | Any,
        *,
        cache_dir: str | Path = ".lecture2pdf-cache",
        total_duration: float | None = None,
        resume: bool = False,
        clear_cache: bool = False,
        stream: Any = None,
        quiet: bool = False,
        verbose: bool = False,
        log_file: str | Path | None = None,
    ) -> "JobContext":
        source = Path(video_path).expanduser().resolve()
        video_fingerprint = fingerprint_file(source)
        config_fingerprint = fingerprint_config(config)
        job_id = job_fingerprint(source, config)[:32]
        cache = make_cache_layout(cache_dir, job_id)
        if clear_cache:
            cache.clear()
            cache = make_cache_layout(cache_dir, job_id)
        checkpoints = CheckpointStore(cache.database)
        checkpoints.initialize(
            video_fingerprint=video_fingerprint,
            config_fingerprint=config_fingerprint,
            job_id=job_id,
        )
        if not resume and checkpoints.latest() is not None:
            # A non-resume invocation starts a clean logical run while keeping
            # the previous job available for explicit --resume.
            checkpoints.save("job", {"status": "restarted"})
        metrics = ProgressMetrics(total_duration=total_duration)
        progress = ProgressReporter(
            metrics=metrics,
            stream=stream,
            quiet=quiet,
            verbose=verbose,
            log_file=log_file,
        )
        return cls(
            video_path=source,
            job_id=job_id,
            video_fingerprint=video_fingerprint,
            config_fingerprint=config_fingerprint,
            cache=cache,
            checkpoints=checkpoints,
            progress=progress,
            interrupts=InterruptController(),
        )

    def checkpoint(self, stage: str, **payload: Any) -> None:
        """Persist the latest consistent stage state before accepting more work."""

        self.checkpoints.save(stage, payload)

    def close(self, *, keep_cache: bool = True) -> None:
        self.progress.close()
        self.checkpoints.close()
        if not keep_cache:
            self.cache.clear()

    def __enter__(self) -> "JobContext":
        self.interrupts.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.interrupts.__exit__(exc_type, exc, traceback)
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            # Persist the last coherent counters even if a caller did not use
            # the lower-level checkpointed_stage helper.
            try:
                self.checkpoints.save(
                    "job",
                    {"status": "interrupted", "progress": self.progress.metrics.snapshot()},
                )
            except Exception:
                # Do not mask the original interruption if the disk is full or
                # the cache directory has become unavailable.
                pass
        self.close()
