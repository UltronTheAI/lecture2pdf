from dataclasses import dataclass

import numpy as np
import pytest

from lecture2pdf.video import Sample, VideoMetadata, sample_frames, validate_metadata
from lecture2pdf.video.decoder import open_decoder


@dataclass
class Frame:
    timestamp: float
    frame_index: int
    frame: object


def test_sampling_uses_timestamps_and_is_bounded():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    frames = [Frame(t, i, image) for i, t in enumerate((0.0, 0.1, 0.49, 0.51, 1.02))]
    samples = list(sample_frames(frames, fps=2))
    assert [sample.time for sample in samples] == [0.0, 0.51, 1.02]
    assert all(isinstance(sample, Sample) for sample in samples)


def test_metadata_validation_rejects_invalid_video(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="dimensions"):
        validate_metadata(VideoMetadata(source, 0, 720, 30, 1))


def test_decoder_selection_validates_backend(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="decoder must be one of"):
        open_decoder(source, backend="cuda")
