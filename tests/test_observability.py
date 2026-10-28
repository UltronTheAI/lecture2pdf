from __future__ import annotations

import io
from pathlib import Path

import pytest

from lecture2pdf.cache import make_cache_layout
from lecture2pdf.checkpoint import CheckpointMismatchError, CheckpointStore
from lecture2pdf.fingerprints import fingerprint_config, fingerprint_file
from lecture2pdf.interrupts import InterruptController, InterruptedRun
from lecture2pdf.job import JobContext
from lecture2pdf.progress import ProgressMetrics, ProgressReporter


def test_fingerprints_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"video data")
    assert fingerprint_config({"b": 2, "a": 1}) == fingerprint_config({"a": 1, "b": 2})
    assert fingerprint_file(source) == fingerprint_file(source)


def test_cache_layout_is_bounded_and_atomic(tmp_path: Path) -> None:
    layout = make_cache_layout(tmp_path / "cache", "job-1")
    target = layout.write_bytes("slides", "slide.png", b"png")
    assert target.read_bytes() == b"png"
    with pytest.raises(ValueError):
        layout.child("slides", "..\\outside.txt")


def test_checkpoint_recovers_latest_and_rejects_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "metadata.sqlite"
    with CheckpointStore(database) as store:
        store.initialize(video_fingerprint="video", config_fingerprint="config", job_id="job")
        saved = store.save("detect", {"frame": 12, "timestamp": 4.0})
        store.mark_completed("slide-1", {"path": "slides/1.png"})
        assert store.latest() == saved
        assert store.is_completed("slide-1")
        with pytest.raises(CheckpointMismatchError):
            store.initialize(video_fingerprint="changed", config_fingerprint="config", job_id="job")


def test_progress_reports_throughput_and_eta() -> None:
    metrics = ProgressMetrics(total_duration=100.0)
    metrics.started_at -= 10.0
    metrics.update(processed_timestamp=50.0, sampled_frames=25, unique_slides=3)
    assert metrics.processing_fps > 0
    assert metrics.eta_seconds is not None
    stream = io.StringIO()
    with ProgressReporter(metrics=metrics, stream=stream, interval_seconds=0) as reporter:
        message = reporter.emit(force=True)
    assert "50.0%" in message
    assert "Slides: 3" in stream.getvalue()


def test_interrupt_controller_is_cooperative() -> None:
    controller = InterruptController()
    controller.requested = True
    with pytest.raises(InterruptedRun):
        controller.raise_if_requested()


def test_job_context_reuses_deterministic_cache(tmp_path: Path) -> None:
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    config = {"sample_fps": 2, "quality": "balanced"}
    first = JobContext.open(video, config, cache_dir=tmp_path / "cache")
    first.checkpoint("detect", frame=4)
    job_id = first.job_id
    first.close()
    resumed = JobContext.open(video, config, cache_dir=tmp_path / "cache", resume=True)
    assert resumed.job_id == job_id
    assert resumed.checkpoints.latest("detect").payload["frame"] == 4
    resumed.close()


def test_job_context_persists_interruption_checkpoint(tmp_path: Path) -> None:
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    context = JobContext.open(video, {"sample_fps": 2}, cache_dir=tmp_path / "cache")
    database = context.cache.database
    with pytest.raises(InterruptedRun):
        with context:
            raise InterruptedRun("stop")
    with CheckpointStore(database) as reopened:
        latest = reopened.latest("job")
        assert latest is not None
        assert latest.payload["status"] == "interrupted"
