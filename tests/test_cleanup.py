from pathlib import Path

from lecture2pdf.cleanup import cleanup_all, cleanup_generated


def test_cleanup_removes_caches_but_preserves_source_and_pdf(tmp_path: Path):
    (tmp_path / ".lecture2pdf-cache" / "jobs" / "abc").mkdir(parents=True)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "lecture_assets").mkdir()
    (tmp_path / "lecture.mp4").write_bytes(b"source")
    (tmp_path / "notes.pdf").write_bytes(b"pdf")
    report = cleanup_generated(tmp_path)
    assert report.removed_count >= 2
    assert not (tmp_path / ".lecture2pdf-cache").exists()
    assert not (tmp_path / "__pycache__").exists()
    assert (tmp_path / "lecture_assets").exists()
    assert (tmp_path / "lecture.mp4").exists()
    assert (tmp_path / "notes.pdf").exists()


def test_cleanup_all_removes_generated_assets(tmp_path: Path):
    (tmp_path / "lecture_assets").mkdir()
    (tmp_path / "benchmark_outputs").mkdir()
    report = cleanup_all(tmp_path)
    assert report.removed_count == 2
    assert not (tmp_path / "lecture_assets").exists()
    assert not (tmp_path / "benchmark_outputs").exists()
