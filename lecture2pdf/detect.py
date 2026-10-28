import cv2
import numpy as np
from skimage.metrics import structural_similarity
from dataclasses import dataclass
from collections import deque
from typing import Optional

def normalized(frame, width=640):
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        raise ValueError("Frame must be a non-empty image")
    h, w = frame.shape[:2]
    nw = min(width, w)
    resized = cv2.resize(frame, (nw, max(1, round(h*nw/w))))
    return resized if resized.ndim == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

def _histogram_distance(a, b):
    ha = cv2.calcHist([a], [0], None, [32], [0, 256])
    hb = cv2.calcHist([b], [0], None, [32], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)))

def _text_regions(gray):
    """Coarse text/line mask used as a cheap OCR-independent signal."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(black_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


@dataclass(frozen=True)
class DetectionMetadata:
    """Serializable explanation for one detector sample."""
    time: float
    kind: str
    confidence: float
    score: float
    threshold: float
    stable_threshold: float
    stable_count: int
    signals: dict

    def as_dict(self):
        return {
            "time": self.time, "kind": self.kind, "confidence": self.confidence,
            "score": self.score, "threshold": self.threshold,
            "stable_threshold": self.stable_threshold,
            "stable_count": self.stable_count, "signals": dict(self.signals),
        }


def compare_signals(a, b, *, content_roi=None):
    """Return explainable multi-signal differences in the [0, 1] range."""
    ga, gb = normalized(a), normalized(b)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    if content_roi is not None:
        x, y, w, h = content_roi
        H, W = ga.shape[:2]
        x0, y0 = max(0, round(x * W)), max(0, round(y * H))
        x1, y1 = min(W, round((x + w) * W)), min(H, round((y + h) * H))
        if x1 > x0 and y1 > y0:
            ga, gb = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
    structural = float(1.0 - structural_similarity(ga, gb, data_range=255))
    ea = cv2.Canny(ga, 60, 140) > 0
    eb = cv2.Canny(gb, 60, 140) > 0
    edge = float(np.mean(ea != eb))
    pixel = float(np.mean(cv2.absdiff(ga, gb) > 35))
    histogram = _histogram_distance(ga, gb)
    # Motion in a small border is more likely to be a lecturer/camera effect;
    # the full-frame score remains useful for real slide transitions.
    border = np.concatenate((ga[: max(1, ga.shape[0] // 12), :].ravel(), ga[-max(1, ga.shape[0] // 12):, :].ravel()))
    border_b = np.concatenate((gb[: max(1, gb.shape[0] // 12), :].ravel(), gb[-max(1, gb.shape[0] // 12):, :].ravel()))
    border_motion = float(np.mean(np.abs(border.astype(np.int16) - border_b.astype(np.int16))) / 255.0)
    text_region = float(np.mean(_text_regions(ga) != _text_regions(gb)))
    hash_distance = float(hamming(phash(ga), phash(gb)) / 64.0)
    combined = float(0.30 * structural + 0.16 * edge + 0.12 * pixel
                    + 0.12 * histogram + 0.16 * hash_distance + 0.10 * text_region
                    + 0.04 * min(1.0, border_motion))
    return {"structural": structural, "edge": edge, "pixel": pixel,
            "histogram": histogram, "border_motion": border_motion,
            "hash_distance": hash_distance, "text_region": text_region,
            "combined": combined}

def difference(a, b):
    """Structural, edge and changed-pixel distances; higher means more different."""
    return compare_signals(a, b)["combined"]

def phash(frame):
    gray = cv2.resize(normalized(frame), (32, 32)).astype(np.float32)
    coeff = cv2.dct(gray)[:8, :8]
    med = np.median(coeff[1:, :])
    return (coeff > med).flatten()

def hamming(a, b):
    return int(np.count_nonzero(a != b))

def is_duplicate(a, b, max_distance=8, max_difference=0.05):
    return hamming(phash(a), phash(b)) <= max_distance and difference(a, b) <= max_difference

class SlideDetector:
    """Stable, adaptive and explainable slide-change detector.

    ``push`` retains the original return value. ``last_decision`` remains a
    dictionary for callers that already consume it, while ``last_metadata``
    provides a typed, serializable record for manifests and diagnostics.
    """
    def __init__(self, change_threshold=0.19, stable_threshold=0.09, min_stable=2,
                 adaptive=True, *, content_roi=None, history_size=31,
                 adaptive_sensitivity=3.5, max_seen=2048, backend=None):
        if change_threshold < 0 or stable_threshold < 0:
            raise ValueError("Detection thresholds must be non-negative")
        if min_stable <= 0 or history_size <= 0 or max_seen <= 0:
            raise ValueError("min_stable, history_size, and max_seen must be positive")
        self.change_threshold = change_threshold
        self.stable_threshold = stable_threshold
        self.min_stable = min_stable
        self.content_roi = content_roi
        self.adaptive_sensitivity = float(adaptive_sensitivity)
        self.max_seen = int(max_seen)
        self.current = None
        self.candidate = None
        self.candidate_time = None
        self.streak = 0
        self.seen = []
        self.history = deque(maxlen=history_size)
        self.adaptive = adaptive
        self.last_decision = None
        self.last_metadata: Optional[DetectionMetadata] = None
        self.backend = backend

    @property
    def effective_change_threshold(self):
        if not self.adaptive or len(self.history) < 5:
            return self.change_threshold
        recent = np.asarray(self.history, dtype=np.float32)[-20:]
        baseline = float(np.median(recent))
        spread = float(np.median(np.abs(recent - baseline)))
        return min(0.65, max(self.change_threshold,
                             baseline + self.adaptive_sensitivity * max(spread, 0.005)))

    @property
    def adaptive_threshold(self):
        """Alias for the current robust threshold used by new integrations."""
        return self.effective_change_threshold

    def _set_decision(self, sample, kind, confidence, score, threshold, stable_count, signals):
        self.last_decision = {
            "kind": kind, "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "signals": signals, "threshold": threshold,
            "stability_samples": stable_count,
        }
        self.last_metadata = DetectionMetadata(
            time=float(sample.time), kind=kind,
            confidence=float(np.clip(confidence, 0.0, 1.0)), score=float(score),
            threshold=float(threshold), stable_threshold=float(self.stable_threshold),
            stable_count=int(stable_count), signals=dict(signals),
        )

    def push_with_metadata(self, sample):
        """Return ``(legacy_decision, DetectionMetadata)`` for one sample."""
        result = self.push(sample)
        return result, self.last_metadata

    def push(self, sample):
        frame = sample.frame
        if self.current is None:
            self.current = frame.copy()
            self.seen.append(frame.copy())
            self._set_decision(sample, "new_slide", 1.0, 1.0,
                               self.effective_change_threshold, 1, {})
            return (sample.time, frame.copy())
        signals = compare_signals(self.current, frame, content_roi=self.content_roi)
        if self.backend is not None and getattr(self.backend, "is_cuda", False):
            # Use a small coarse image to keep transfer overhead bounded;
            # this is an additional GPU signal, not a silent CPU fallback.
            coarse_current = cv2.resize(self.current, (320, 180), interpolation=cv2.INTER_AREA)
            coarse_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            signals["gpu_pixel"] = float(self.backend.difference(coarse_current, coarse_frame))
        else:
            signals["gpu_pixel"] = None
        delta = signals["combined"]
        if signals["gpu_pixel"] is not None:
            delta = 0.90 * delta + 0.10 * signals["gpu_pixel"]
            signals["combined_with_gpu"] = delta
        threshold = self.effective_change_threshold
        # A large border-only movement is commonly the presenter or camera;
        # require more stable evidence before calling it a new slide.
        if delta < threshold or (signals["border_motion"] > 0.24 and delta < threshold * 1.35):
            # Only stable same-slide observations are valid baseline samples;
            # candidate transitions must not raise the future change threshold.
            self.history.append(delta)
            self.candidate = None
            self.streak = 0
            self._set_decision(sample, "same_slide", max(0.0, 1.0 - delta), delta,
                               threshold, 0, signals)
            return None
        if (self.candidate is not None
                and compare_signals(self.candidate, frame, content_roi=self.content_roi)["combined"] < self.stable_threshold):
            self.streak += 1
        else:
            self.candidate = frame.copy()
            self.candidate_time = sample.time
            self.streak = 1
        if self.streak >= self.min_stable:
            chosen = self.candidate
            chosen_time = self.candidate_time
            self.current = chosen.copy()
            self.candidate = None
            self.streak = 0
            # Respect a caller's stricter transition threshold when deciding
            # whether a stabilized candidate is genuinely a duplicate.
            duplicate_limit = min(0.14, threshold)
            if any(is_duplicate(chosen, old, max_difference=duplicate_limit) for old in self.seen):
                self._set_decision(sample, "duplicate", 1.0 - min(1.0, delta), delta,
                                   threshold, self.min_stable, signals)
                return None
            self.seen.append(chosen.copy())
            if len(self.seen) > self.max_seen:
                del self.seen[:len(self.seen) - self.max_seen]
            confidence = min(1.0, max(0.0, delta / max(threshold, 1e-6)))
            self._set_decision(sample, "new_slide", confidence, delta, threshold,
                               self.min_stable, signals)
            return (chosen_time, chosen.copy())
        self._set_decision(sample, "candidate",
                           min(1.0, delta / max(threshold, 1e-6)), delta,
                           threshold, self.streak, signals)
        return None
