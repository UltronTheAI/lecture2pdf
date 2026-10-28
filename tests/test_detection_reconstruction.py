import cv2
import numpy as np

from lecture2pdf.detect import SlideDetector, compare_signals
from lecture2pdf.reconstruct import reconstruct, reconstruct_with_metadata
from lecture2pdf.video import Sample


def _slide(label, border=0):
    image = np.full((180, 320, 3), 30, dtype=np.uint8)
    cv2.putText(image, label, (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (245, 245, 245), 3)
    if border:
        image[:, :18] = border
    return image


def test_compare_signals_exposes_independent_explainable_signals():
    signals = compare_signals(_slide("A"), _slide("B"))
    assert signals["combined"] > 0
    for key in ("structural", "edge", "pixel", "histogram", "hash_distance", "text_region"):
        assert key in signals
        assert 0 <= signals[key] <= 1


def test_detector_metadata_reports_stability_and_reason():
    detector = SlideDetector(change_threshold=0.005, stable_threshold=0.03, min_stable=2)
    first = detector.push_with_metadata(Sample(0.0, _slide("A")))
    assert first[0] is not None
    assert first[1].kind == "new_slide"

    detector.push_with_metadata(Sample(1.0, 255 - _slide("B")))
    assert detector.last_metadata.kind == "candidate"
    decision, metadata = detector.push_with_metadata(Sample(2.0, 255 - _slide("B")))
    assert decision is not None
    assert metadata.kind == "new_slide"
    assert metadata.confidence > 0
    assert metadata.signals["structural"] > 0
    assert metadata.as_dict()["stable_count"] == 2


def test_detector_adapts_to_repeated_low_level_motion():
    detector = SlideDetector(change_threshold=0.01, history_size=8)
    base = _slide("A")
    detector.push(Sample(0.0, base))
    for i in range(8):
        noisy = np.clip(base.astype(np.int16) + (i % 2) * 3, 0, 255).astype(np.uint8)
        detector.push(Sample(i + 1.0, noisy))
    assert detector.adaptive_threshold >= detector.change_threshold
    assert len(detector.history) <= 8


def test_reconstruction_is_bounded_and_reports_uncertainty():
    frames = (_slide("A", border=(i % 3) * 80) for i in range(100))
    result = reconstruct_with_metadata(frames, max_frames=5)
    assert result["observed_frames"] == 5
    assert result["fabricated"] is False
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["uncertain_fraction"] <= 1


def test_reconstruction_selects_observed_values(monkeypatch):
    import lecture2pdf.reconstruct as reconstruction

    def identity(reference, frame):
        return frame.copy(), np.ones(reference.shape[:2], dtype=bool), True

    monkeypatch.setattr(reconstruction, "_align_with_mask", identity)
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full_like(a, 100)
    c = np.full_like(a, 255)
    output = reconstruct([a, b, c], max_frames=3)
    assert set(np.unique(output).tolist()).issubset({0, 100, 255})

